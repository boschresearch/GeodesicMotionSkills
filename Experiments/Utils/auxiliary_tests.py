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


import torch
import numpy as np
import pickle
from sklearn.utils.extmath import cartesian


class Tests:
    """Centralized test class, implementing experiments such as geodesic computation between
    arbitrary points in the latent space
    """

    def __init__(self):

        self.point_dim = 100  # number of points in each dimension (latent grid) for metric visualization
        self.pos_dim = 2  # r2 data dimension in euclidean space
        self.qua_dim = 3  # s2 data dimension on surface on the 2-hypersphere
        self.latent_max = 10  # metric visualization max/min value. should be equal to Latent_max in vae_toy_example.py

        self.decoded_grid = None
        self.mf = None
        self.measure = None
        self.model = None

        # define latent grid X x Y
        x = np.linspace(-self.latent_max, self.latent_max, self.point_dim)
        y = np.linspace(-self.latent_max, self.latent_max, self.point_dim)
        self.latent_grid = cartesian((x.flatten(), y.flatten()))

        # define arbitrary points for geodesic computation
        self.geodesic_start = torch.tensor([0, -.3])
        self.geodesic_end = torch.tensor([-.3, 0])

    def geodesic_computation(self, model, ref_trajectory, discrete_model=None):
        """Computing geodesic between two points on the manifold
        Input:
            model:              an instance of the class Toy_example.VAE()
            ref_trajectory:     a reference trajectory selected from the test_set/training_set
                                (It can be any trajectory but it has been selected from training
                                set to make sure the points are already on the manifold)
            discrete_model:     a discrete graph representing the manifold
        output:
            measure:            variance measure
            mf                  magnification factor
            latent_curves       geodesic in latent space
        """
        print("Compute geodesic...")
        self.model = model
        self.model.eval()

        # the goal index of the target point on the reference trajectory
        goal_index = -1
        p0 = self.model.encode(torch.tensor(ref_trajectory[0][0]).float(), train_rbf=True)[1]
        p1 = self.model.encode(torch.tensor(ref_trajectory[0][goal_index]).float(), train_rbf=True)[1]

        # Compute geodesic between p0 and p1
        curve = discrete_model.connecting_geodesic(p0.view(1, -1), p1.view(1, -1), self.model, self.model.time_step)

        print("Compute visualization data...")
        alpha = torch.linspace(0, 1, 500, device=curve.device).reshape((-1, 1))
        latent_curves = curve(alpha.transpose(1, 0)).detach().numpy()  # Latent geodesic
        embedded_curves = latent_curves[:, 1:] ** 2
        optimized_curves = self.model.embed(curve(alpha.transpose(1, 0)), jacobian=False).detach().numpy()

        # store the trajectory for future tests on the robot
        alpha = torch.linspace(0, 1, 500, device=curve.device).reshape((-1, 1))
        positions = self.model.decode(curve(alpha.transpose(1, 0)), train_rbf=True)[0]
        points_on_curve_r2 = positions.mean.detach().numpy()
        pickle.dump(points_on_curve_r2, open("../../Trajectory_R2.p", "wb"), protocol=2)

        # store the trajectory for future tests on the robot
        quaternions = self.model.decode(curve(alpha.transpose(1, 0)), train_rbf=True)[1]
        points_on_curve_s2 = quaternions.loc.detach().numpy()
        pickle.dump(points_on_curve_s2, open("../../Trajectory_S2.p", "wb"), protocol=2)

        #  calculating decoder Jacobian for all the points on the latent grid
        metrics = model.embed(torch.tensor(self.latent_grid).float().unsqueeze_(0), True)
        self.mf = np.array([np.log(np.sqrt(np.abs(np.linalg.det(metrics[2][i])))) for i in
                  range(self.point_dim * self.point_dim)]).reshape(self.point_dim, self.point_dim)
        self.measure = np.array([np.sum(metrics[0][0, i, 5:].detach().numpy()) for i in
                                 range(self.point_dim * self.point_dim)]).reshape(self.point_dim, self.point_dim)
        self.measure = np.log(self.measure)

        return self.measure, self.mf, latent_curves
