""" Learning Riemannian Manifolds for Geodesic Motion Skills (R:SS 2021).
Copyright (c) 2022 Robert Bosch GmbH

@author: Hadi Beik-Mohammadi
@author: Leonel Rozo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

import matplotlib.pyplot as plt
import numpy as np
import hyperspherical_vae.distributions.von_mises_fisher as vmf
import torch
import torch.distributions as td
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import glob
import pickle
from GeodesicMotionSkills.Experiments.Utils import auxiliary_tests, discretized_manifold
from GeodesicMotionSkills.Experiments.Utils.environment import Environment
from stochman.manifold import EmbeddedManifold
from stochman import nnj
import copy


class VAE(nn.Module, EmbeddedManifold):

    def __init__(self, layers, batch_size, sigma=1e-6, sigma_z=0.1):
        """
        Create a Variational Auto-Encoder (VAE) neural network integrated with an embedded Riemannian manifold.
            Input:
                layers:     number of neurons in each layer of the VAE's encoder [D, h_1, ..., h_n, d]
                            (D: Ambient space dimension, h_n: nth hidden layer size,
                            d: latent space dimension, tested with 2 dimensions)
                            the reverse is used for the decoder [d, h_n, ..., h_1, D].
                batch_size: training batch size
                sigma:      the scale parameter of the distribution given as the output of the VAE's decoder
                            for position data
                sigma_z:    the scale parameter of the distribution given as the output of the VAE's encoder
        """
        super(VAE, self).__init__()
        self.tests = None
        self.env = Environment()

        self.p = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers[1:-1]  # Dimension of hidden layers
        self.obstacle_input_space = None

        # Hyper-parameters
        self.plot_number = "0"  # Automatically Set
        self.device = 'cpu'
        self.batch_size = batch_size
        self.kl_coeff = 1.0  # Automatically Set
        self.kl_coeff_max = 1.0
        self.obstacle_radius = 0.02
        self.num_clusters = 500  # Number of clusters in the RBF k_mean
        self.quaternion_log_scale = 1e10  # the scale of quaternion log-likelihood in the ELBO
        self.vmf_concentration_scale = 1e2  # the scale of vmf distribution concentration

        #  Flags
        self.empowered_quaternions = False  # when True, it will scale the the LogLik... of the quaternions in the Loss
        self.activate_KL = False  # if enables KL is considered in the ELBO calculation
        self.visualization = False
        self.train_var = False  # Automatically Set

        self.time_step = 0
        out_features = None

        #  Initialize VAE
        enc = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.BatchNorm1d(in_features))
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        enc.append(nnj.Linear(out_features, self.d))

        enc_scale = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc_scale.append(nnj.BatchNorm1d(in_features))
            enc_scale.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features), nnj.Softplus()))

        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            dec.append(nnj.BatchNorm1d(in_features))
            dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        dec.pop(0)  # remove initial batch-norm as it serves no purpose
        dec.append(nnj.Linear(out_features, self.p))

        self.encoder_loc = nnj.Sequential(*enc)
        self.decoder_loc = nnj.Sequential(*dec)
        self.encoder_scale = nnj.Sequential(*enc_scale)

        self.encoder_scale_fixed = nn.Parameter(torch.tensor([sigma_z]), requires_grad=False)
        self.decoder_scale_pos = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.decoder_scale_qua = nn.Parameter(torch.tensor(np.ones((self.batch_size, 3)) *
                                                           self.vmf_concentration_scale), requires_grad=False)
        self.dec_std_pos = lambda z: torch.ones(20, self.p, device=self.device)
        self.dec_std_qua = lambda z: torch.ones(20, self.p, device=self.device)

        self.prior_loc = nn.Parameter(torch.zeros(self.d), requires_grad=False)
        self.prior_scale = nn.Parameter(torch.ones(self.d), requires_grad=False)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)

    def embed(self, points, jacobian=False):
        """
        Embed the manifold into (mu, std) space.

        Input:
            points:     a Nx(d) or BxNx(d) torch Tensor representing a (batch of a)
                        set of N points in latent space that will be embedded
                        in R^2D.

        Optional input:
            jacobian:   a boolean indicating if the Jacobian matrix of the function
                        should also be returned. Default is False.

        Output:
            embedded:   a Nx(2D) of BxNx(2D) torch tensor containing the N embedded points.
                        The first Nx(d) part contain the mean part of the embedding,
                        whlie the last Nx(d) part contain the standard deviation
                        embedding.

        Optional output:
            J:          If jacobian=True then a second Nx(2D)x(d) or BxNx(2D)x(d)
                        torch tensor is returned that contain the Jacobian matrix
                        of the embedding function.
        """
        std_scale = 1.0
        metric = None
        j = None
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD
        if jacobian:
            mu_pos, mu_qua, j_mu = self.decode(points, train_rbf=True, jacobian=True)  # BxNxD, BxNxDx(d)
            std, j_std = self.dec_std_pos(points, jacobian=True)  # BxNxD, BxNxDx(d)
            std_qua, j_std_qua = self.dec_std_qua(points, jacobian=True)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu_pos.mean, mu_qua.loc.unsqueeze(0), std_scale * std, std_scale * (1 / std_qua)),
                                 dim=2)  # BxNx(2D)
            j = torch.cat((j_mu, torch.cat((std_scale * j_std.squeeze(0), std_scale * j_std_qua.squeeze(0)), dim=1)),
                          dim=2)  # BxNx(2D)x(d)
            m = torch.einsum("bji,bjk->bik", j_mu, j_mu)
            m2 = torch.einsum("bji,bjk->bik", j_std.squeeze(0), j_std.squeeze(0))
            m3 = torch.einsum("bji,bjk->bik", j_std_qua.squeeze(0), j_std_qua.squeeze(0))
            metric = (m3 + m2 + m).detach().numpy()
        else:
            mu_pos, mu_qua = self.decode(points, train_rbf=True, jacobian=False)  # BxNxD, BxNxDx(d)
            std = self.dec_std_pos(points, jacobian=False)  # BxNxD, BxNxDx(d)
            std_qua = self.dec_std_qua(points, jacobian=False)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu_pos.mean, mu_qua.loc.unsqueeze(0), std_scale * std, std_scale * (1 / std_qua)),
                                 dim=2)  # BxNx(2D)
        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                j = j.squeeze(0)
        if jacobian:
            return embedded, j, metric
        else:
            return embedded

    def encode(self, x, train_rbf=False):
        """ Encode the input space sample
        Inputs:
            x: a torch Tensor corresponding to one input space point.
            train_rbf: True when training the RBFs
        Outputs:
            z_distribution: latent space encoded distribution given x
        """
        z_loc = self.encoder_loc(x)
        if train_rbf:
            z_scale = self.encoder_scale(x)
        else:
            z_scale = self.encoder_scale_fixed
        z_distribution = td.Independent(td.Normal(loc=z_loc, scale=z_scale, validate_args=False), 1), z_loc
        return z_distribution

    def decode(self, z, train_rbf=False, jacobian=False, negative=False):
        """ compute the input space estimation given the latent variable
        Inputs:
            z: sample from latent space
            train_rbf: True when training the RBFs
            jacobian: True to calculate Jacobian
            negative: True to flip the quaternion
        Outputs:
            position_distribution: gaussian distribution given z
            quaternion_distribution: vMF distribution given z
        """
        # Since batch normalization is a bit of a mess we have to apply
        # a series of reshape's to get the correct behavior
        quaternion_distribution_negative = None  # used when p(x|z) = ½vMF(x|mu(z), k(z)) + ½vMF(x|-mu(z), k(z))
        ja = None
        if jacobian:
            x_loc, ja = self.decoder_loc(z.view(-1, self.d), jacobian=jacobian)
        else:
            x_loc = self.decoder_loc(z.view(-1, self.d))
        position_scale = self.decoder_scale_pos + 1e-10
        quaternion_scale = self.decoder_scale_qua + 1e-10

        x_var_pos = self.dec_std_pos(z.view(-1, self.d))
        x_var_qua = self.dec_std_qua(z.view(-1, self.d))

        position_loc = x_loc[:, :pos_dof]
        quaternion_loc = x_loc[:, pos_dof:]
        qua_mean = quaternion_loc / quaternion_loc.norm(dim=-1, keepdim=True)

        x_shape = list(z.shape)
        x_shape[-1] = position_loc.shape[-1]

        quaternion_distribution = vmf.VonMisesFisher(qua_mean, quaternion_scale)
        quaternion_distribution_negative = vmf.VonMisesFisher(-qua_mean, quaternion_scale)
        position_distribution = td.Independent(
            td.Normal(loc=position_loc.view(torch.Size(x_shape)), scale=position_scale), 1)
        if train_rbf:
            if negative:
                quaternion_distribution_negative = vmf.VonMisesFisher(-qua_mean, x_var_qua)
            quaternion_distribution = vmf.VonMisesFisher(qua_mean, x_var_qua)

            position_distribution = td.Independent(
                td.Normal(loc=position_loc.view(torch.Size(x_shape)), scale=x_var_pos), 1)

        if jacobian:
            return position_distribution, quaternion_distribution, ja
        if negative:
            return position_distribution, quaternion_distribution, quaternion_distribution_negative
        return position_distribution, quaternion_distribution

    def disable_training(self):
        """ Disabling the training for all the networks
        Inputs:

        Outputs:

        """
        for module in self.encoder_loc._modules.values():
            module.training = False
        for module in self.decoder_loc._modules.values():
            module.training = False
        # self.encoder_loc.disable_training()
        # self.decoder_loc.disable_training()

    def init_std(self, x, load_clusters=False):
        """ initializing the RBF networks
        Inputs:
            x: a torch Tensor corresponding to one input space point.
            load_clusters: loading the clusters from a file
        Outputs:
            cluster_centers: center of clusters computed by kmeans
        """
        self.train_var = True
        with torch.no_grad():
            _, z = self.encode(x, train_rbf=True)
        d = z.shape[1]
        inv_max_std = np.sqrt(1e-12)  # 1.0 / x.std()
        beta = 10.0 / z.std(dim=0).mean()  # 1.0
        rbf_beta = beta * torch.ones(1, self.num_clusters)
        k_means = KMeans(n_clusters=self.num_clusters).fit(z.numpy())
        if load_clusters:
            k_means = pickle.load(open("../Clusters/" + "clusters.p", "rb"))
        else:
            pickle.dump(k_means, open("../Clusters/" + "clusters.p", "wb"))
        centers = torch.tensor(k_means.cluster_centers_)
        self.dec_std_pos = nnj.Sequential(nnj.RBF(d, self.num_clusters, points=centers, beta=rbf_beta),
                                          # d --> num_clusters
                                          nnj.PosLinear(self.num_clusters, 1, bias=False),  # num_clusters --> 1
                                          nnj.Reciprocal(inv_max_std),  # 1 --> 1
                                          nnj.PosLinear(1, pos_dof))  # 1 --> D
        self.dec_std_qua = nnj.Sequential(nnj.RBF(d, self.num_clusters, points=centers, beta=rbf_beta),
                                          # d --> num_clusters
                                          nnj.PosLinear(self.num_clusters, qua_dof),  # num_clusters --> 1
                                          )  # 1 --> D
        self.dec_std_pos.to(self.device)
        self.dec_std_qua.to(self.device)
        cluster_centers = k_means.cluster_centers_
        return cluster_centers

    def fit_std(self, data_loader, num_epochs, model):
        """ Training the standard deviation models including 2 RBF networks for Cartesian and quaternion decoders
        Inputs:
            data_loader: the demonstration training dataset
            num_epochs: number of epochs
            model: an instance of VAE class
            loss_list: list of previous computed losses
            loss_list_KL: list of previous computed KL
            loss_list_log: list of previous computed log likelihoods
            n_samples: number of samples used to calculate reconstruction error
        Outputs:

        """
        params = list(self.encoder_scale.parameters()) + list(self.dec_std_qua.parameters()) + list(
            self.dec_std_pos.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        for epoch in range(num_epochs):
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                loss, loss_kl, loss_log = loss_function_elbo(data, model, train_rbf=True, n_samples=n_samples)
                loss.backward()
            optimizer.step()
            print('Training RBF Networks ====> Epoch: {}/{}'.format(epoch, epochs))


def loss_function_elbo(x, model, train_rbf, n_samples):
    """ Loss function computing evidence lower bound
    Inputs:
        x: a torch Tensor corresponding to one input space point.
        model: an instance of VAE class
        train_rbf: enable when training the RBF network
        n_samples: number of samples used to calculate reconstruction error
    Outputs:
        elbo: Evidence lower bound 
        KL: Kullback-leibler divergence
        log_mean: mean over log likelihood
    """
    q, _ = model.encode(x, train_rbf=train_rbf)

    z = q.rsample(torch.Size([n_samples]))  # (n_samples)x(batch size)x(latent dim)
    px_z, px_qua_z, px_qua_z_n = model.decode(z, train_rbf=train_rbf, negative=True)  # p(x|z)

    log_p_negative = log_prob(model, x, px_z, px_qua_z, px_qua_z_n)  # vMF(x|mu(z), k(z))
    log_p_negative = torch.mean(log_p_negative) * 1

    log_p = log_p_negative
    kl = torch.tensor([0.0])
    if model.activate_KL:
        log_p = log_p_negative
        kl = -0.5 * torch.sum(1 + q.variance.log() - q.mean.pow(2) - q.variance) * model.kl_coeff
        kl = kl * 400000000
        elbo = torch.mean(log_p - kl, dim=0)
    else:
        elbo = torch.mean(log_p, dim=0)
    log_mean = torch.mean(log_p, dim=0)
    return -elbo, kl, log_mean


def train(model, optimizer, loss_function, data_loader, epoch, device, train_rbf):
    """ Training the model
    Inputs:
        model: an instance of VAE class
        optimizer: an instance of optimizer
        loss_function: an instance of loss_function function
        data_loader: the demonstration training dataset
        epoch: current epoch of learning session
        device: allocated device to pytorch
        train_rbf: enable when training the RBF network
    Outputs:
        avg_loss: the average loss over the training data
        avg_loss_KL: the average KL loss over the training data
        avg_loss_KL: the average log likelihood over the training data
        dataset_length: the dataset length
    """
    model.train()
    for batch_idx, (data,) in enumerate(data_loader):
        data = data.to(device)
        # prevent crashing when the leftover training data is not enough got an epoch
        if data.shape[0] != batch_size:
            break
        optimizer.zero_grad()
        batch_loss, loss_kl, loss_log = loss_function(data, train_rbf)
        batch_loss.backward()
        optimizer.step()
    print('Training ====> Epoch: {}/{}'.format(epoch, epochs))


def train_model():
    """ Training all the networks in 3 step each focused on regularization, reconstruction and solely training RBF
    Inputs:
        none
    Outputs:
        none
    """
    for encoder_scale in encoder_scales:
        for repetition in range(repetitions):
            model = VAE(layers=[dof, 200, 100, 2], batch_size=batch_size, sigma_z=encoder_scale).to(device)

            # regularization focused training
            model.activate_KL = True
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_function = lambda data, train_rbf: loss_function_elbo(data, model, train_rbf, n_samples=n_samples)
            for epoch in range(int(epochs)):
                train(model, optimizer, loss_function, train_loader, epoch, device, train_rbf=False)

            # Focusing on reconstruction of the data
            model.activate_KL = False
            model.kl_coeff = 0.1
            params = list(model.decoder_loc.parameters())
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            for epoch in range(int(epochs)):
                if epoch == int(epochs / 2):
                    model.empowered_quaternions = True
                train(model, optimizer, loss_function, train_loader, epoch, device, train_rbf=False)
            model.empowered_quaternions = False

            # Train RBF/Variance networks
            model.init_std(train_data.tensors[0].float(), load_clusters=False)
            model.fit_std(train_loader, epochs_rbf, model)

            # Saving the model into a file
            fn = model_path + '/vae_loss%s_ns%d_es%f_r%d.pt' % (loss, n_samples, encoder_scale, repetition)
            print('Saving model:', fn)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.to('cpu').state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'repetition': repetition,
                        'encoder_scale': encoder_scale}, fn)


def test_model():
    """ Testing the learned Riemannian manifold by evaluating the metric and geodesic computation
    Inputs:
        none
    Outputs:
        none
    """
    files = glob.glob(model_path + '/*.pt')

    # Load the torch model
    for fn in files:
        model = VAE(layers=[dof, 200, 100, 2], batch_size=batch_size, sigma_z=encoder_scales[0]).to(device)
        model.obstacle_input_space = None
        model.init_std(train_data.tensors[0].float(), load_clusters=True)
        checkpoint = torch.load(fn)
        model.load_state_dict(checkpoint['model_state_dict'])
        encoder_scale = checkpoint['encoder_scale']
        model.to(device)
        model.disable_training()

    # Graph based manifold generation
    ran = torch.linspace(-latent_max, latent_max, graph_size)
    x, y = torch.meshgrid(ran, ran)
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)))
    print("Compute graph-based manifold ...")
    discrete_model = discretized_manifold.DiscretizedManifold(model, grid)

    tests = auxiliary_tests.Tests()

    var_measure, mag_fac, geodesic = tests.geodesic_computation(model, [test_traj], discrete_model=discrete_model)

    plot(model, var_measure, mag_fac, geodesic)
    model.env.step()


def log_prob(model, x, positional_dist, quaternion_dist, quaternion_dist_n):
    """Compute the log probability of quaternion vmf and position Guassian distributions
    Inputs:
        model: an instace of VAE class
        x: a torch Tensor corresponding to one input space point.
        positional_dist: Guassian distributation related to the Cartesian position.
        quaternion_dist: von Mises-Fisher distribution related to the Quaternion data.
        quaternion_dist_n: flipped/negative mean von Mises-Fisher distribution related to the Quaternion data.
    Outputs:
        curve: summed log-likelihood of position and orientation
    """
    log_p = torch.mean(positional_dist.log_prob(x[:, :pos_dof]), dim=1)
    log_q = torch.mean(quaternion_dist.log_prob(x[:, pos_dof:]), dim=0)
    log_q_n = torch.mean(quaternion_dist_n.log_prob(x[:, pos_dof:]), dim=0)
    log_q = torch.log(torch.clamp((torch.exp(log_q) + torch.exp(log_q_n)) / 2, 1e-10, np.inf))
    log_likelihood = log_p + (model.quaternion_log_scale * log_q)
    return log_likelihood


def plot(model, measure, mf, geodesic):
    """ Plotting the metric
    Inputs:
        model: an instace of VAE class
        measure: variance measure
        mf: magnification factor
        geodesic: geodesic in latent space
    """
    print("Plotting the metric...")

    fig, ax = plt.subplots(figsize=(13, 6))

    ax1 = fig.add_subplot(121)
    ax1.imshow(np.rot90(measure), interpolation='bicubic', extent=[-latent_max, latent_max, -latent_max, latent_max])

    data_points_reconstructed = np.zeros((len(train_loader.dataset), 2))
    for data_index in range(len(train_loader.dataset[:])):
        data_points = train_loader.dataset[:][data_index][0].detach().numpy()
        encoded = model.encode(torch.tensor(data_points), train_rbf=True)
        data_points_reconstructed[data_index] = encoded[0].mean.detach().numpy()

    ax1.scatter(data_points_reconstructed[:, 0], data_points_reconstructed[:, 1], marker=".", color="white", s=200,
                label='Encoded Demonstrations', alpha=0.1)
    ax1.plot(geodesic[0, :, 0], geodesic[0, :, 1], color="yellow")
    
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # tiscks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    ax1.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    ax1.grid(False)
    ax1.legend(fontsize=18, loc='upper right')
    ax1.set_xlim(-latent_max, latent_max)
    ax1.set_ylim(-latent_max, latent_max)
    fig.tight_layout()

    ax1.tick_params(axis='both', which='major', labelsize=17)

    ax2 = fig.add_subplot(122)
    ax2.imshow(np.rot90(mf), interpolation='bicubic',
               extent=[-latent_max, latent_max, -latent_max, latent_max])

    data_points_reconstructed = np.zeros((len(train_loader.dataset), 2))
    for data_index in range(len(train_loader.dataset[:])):
        data_points = train_loader.dataset[:][data_index][0].detach().numpy()
        encoded = model.encode(torch.tensor(data_points), train_rbf=True)
        data_points_reconstructed[data_index] = encoded[0].mean.detach().numpy()

    ax2.scatter(data_points_reconstructed[:, 0], data_points_reconstructed[:, 1], marker=".", color="white", s=200,
                label='Encoded Demonstrations', alpha=0.1)
    ax2.plot(geodesic[0, :, 0], geodesic[0, :, 1], color="yellow")

    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax2.set_xlim(-latent_max, latent_max)
    ax2.set_ylim(-latent_max, latent_max)
    fig.tight_layout()
    ax2.grid(False)
    plt.show()


if __name__ == "__main__":
    # train = 0, visualization = 1
    mode = 1

    trajectory_number = 7  # the number of total demonstration files
    test_id = 0  # the index of the demonstration used for testing
    n_samples = 1
    epochs = 1000  # training Mean
    epochs_rbf = 1000  # training Variance
    repetitions = 1  # number of training
    encoder_scales = [1.0]  # predefined variance of the encoder
    learning_rate = 1e-3  # learning rate
    graph_size = 100  # the number of graph nodes in each dimension
    dof = 5  # number of dimensions (input output vector size)
    pos_dof = 2  # number of dimensions in the position data
    qua_dof = 3  # number of dimensions in the orientation data
    latent_max = 10  # metric visualization max/min. should be equal to Latent_max in auxilary_tests_toy_example.py
    batch_size = 128  # training batch size
    trajectory_length = 135  # number of points in each trajectory

    r2_letter = "J"
    s2_letter = "C"
    loss = "elbo"
    model_path = '../models'
    device = 'cpu'
    input_data = np.zeros((trajectory_number, trajectory_length))
    name = "_delete_no_obstacles"
    trajectory_flatten = None
    plot_num = "0"

    for i in range(trajectory_number):
        r2_file_test = "../Dataset/letter_" + r2_letter + "_R2_" + str(test_id) + ".p"
        r2_file = "../Dataset/letter_" + r2_letter + "_R2_" + str(i) + ".p"
        s2_file_test = "../Dataset/letter_" + s2_letter + "_S2_" + str(test_id) + ".p"
        s2_file = "../Dataset/letter_" + s2_letter + "_S2_" + str(i) + ".p"

        test_traj = pickle.load(open(r2_file, "rb"), encoding="latin1").transpose()
        trajectory = pickle.load(open(r2_file, "rb"), encoding="latin1").transpose()
        test_trajectory_qua = -pickle.load(open(s2_file_test, "rb"), encoding="latin1")
        trajectory_qua = pickle.load(open(s2_file, "rb"), encoding="latin1")

        trajectory_n = copy.deepcopy(trajectory)
        trajectory = np.append(trajectory, trajectory_qua, 1)
        trajectory_n = np.append(trajectory_n, -trajectory_qua, 1)
        test_traj = np.append(test_traj, test_trajectory_qua, 1)

        if i > 0:
            trajectory_flatten = np.vstack([trajectory_flatten, trajectory])
            trajectory_flatten = np.vstack([trajectory_flatten, trajectory_n])  # -Q for training
        else:
            trajectory_flatten = np.array(trajectory)
            trajectory_flatten = np.vstack([trajectory_flatten, trajectory_n])  # -Q for training

    input_data = trajectory_flatten[:, 0:dof]

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(input_data).float())
    x_train, x_test = train_test_split(train_data, test_size=0.3)

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=True)

    if mode == 0:
        train_model()
    elif mode == 1:
        test_model()
