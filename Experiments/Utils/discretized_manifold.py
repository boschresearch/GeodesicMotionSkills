""" Learning Riemannian Manifolds for Geodesic Motion Skills (R:SS 2021).
Copyright (c) 2022 Robert Bosch GmbH

@author: Soren Hauberg (DTU)
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

This is a derived work from class DiscretizedManifold(Manifold): 
developed by Soren Hauberg (DTU) and Nicki Skafte Detlefsen 
and reseased under the Apache 2.0 License. 
cf. https://github.com/MachineLearningLifeScience/stochman/blob/master/stochman/discretized_manifold.py

"""

#!/usr/bin/env python3
import torch
from stochman import *
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time

class DiscretizedManifold:
    def __init__(self, model, grid, use_diagonals=False):
        """Approximate the latent space of a Manifold with a discrete grid.

        Inputs:
            model:  the Manifold to be approximated. This object should
                    implement the 'curve_length' function.

            grid:   a torch Tensor where the first dimension correspond
                    to the latent dimension of the manifold, and the
                    remaining are grid positions in a meshgrid format.
                    For example, a 2D manifold should be discretized by
                    a 2xNxM grid.
        """
        self.grid = grid
        self.G = nx.Graph()

        if grid.shape[0] != 2:
            raise Exception('Currently we only support 2D grids -- sorry!')

        # Add nodes to graph
        dim, xsize, ysize = grid.shape
        node_idx = lambda x, y: x*ysize + y
        self.G.add_nodes_from(range(xsize*ysize))
        
        # add edges
        line = CubicSpline(begin=torch.zeros(1, dim), end=torch.ones(1, dim), num_nodes=2)
        t = torch.linspace(0, 1, 5)
        self.fixed_positions = {}
        self.decoded_positions = torch.zeros((xsize * ysize, 2))

        for x in range(xsize):
            for y in range(ysize):
                line.begin = grid[:, x, y].view(1, -1)

                n = node_idx(x, y)
                self.fixed_positions[n] = (grid[0, x, y].detach().numpy(),grid[1, x, y].detach().numpy())
                self.decoded_positions[n] = model.decode(grid[:, x, y], train_rbf = True)[0].mean.flatten()

                with torch.no_grad():
                    if x > 0:
                        line.end = grid[:, x-1, y].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x-1, y), weight=w)
                    if y > 0:
                        line.end = grid[:, x, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x, y-1), weight=w)
                    if x < xsize-1:
                        line.end = grid[:, x+1, y].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x+1, y), weight=w)
                    if y < ysize-1:
                        line.end = grid[:, x, y+1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x, y+1), weight=w)
                    if use_diagonals and x > 0 and y > 0:
                        line.end = grid[:, x-1, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x-1, y-1), weight=w)
                    if use_diagonals and x < xsize-1 and y > 0:
                        line.end = grid[:, x+1, y-1].view(1, -1)
                        w = model.curve_length(line(t)).item()
                        self.G.add_edge(n, node_idx(x+1, y-1), weight=w)

        self.mem_g = copy.deepcopy(self.G)

    def draw_graph(self, graph_id, curve):
        plt.figure(1, figsize=(12, 12))
        alpha = torch.linspace(0, 1, 500, device=curve.device).reshape((-1, 1))
        latent_curves = curve(alpha).detach().numpy()

        pos = nx.spring_layout(self.G, fixed=self.G.nodes.keys(), pos=self.fixed_positions)
        nx.draw_networkx_nodes(self.G, pos, node_size=5)
        # edges
        elarge = [(u, v) for (u, v, d) in self.G.edges(data=True)]
        weights = [(0, 0, 0, d['weight']) for (u, v, d) in self.G.edges(data=True)]

        nx.draw_networkx_edges(self.G, pos, edgelist=elarge, edge_color=weights, width=4, edge_cmap=plt.cm.Blues)
        # nx.draw_networkx_edges(self.G, pos, edgelist=esmall, width=2, alpha=0.5, edge_color="b", style="dashed")
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], edgecolors="yellow")
        plt.plot(latent_curves[:, 0], latent_curves[:, 1])
        plt.axis("off")
        plt.show()

    def grid_point(self, p):
        """Return the index of the nearest grid point.

        Input:
            p:      a torch Tensor corresponding to a latent point.
        
        Output:
            idx:    an integer correponding to the node index of
                    the nearest point on the grid.
        """
        return (self.grid.view(self.grid.shape[0], -1) - p.view(-1, 1)).pow(2).sum(dim=0).argmin().item()

    def shortest_path(self, p1, p2):
        """Compute the shortest path on the discretized manifold.

        Inputs:
            p1:     a torch Tensor corresponding to one latent point.

            p2:     a torch Tensor corresponding to another latent point.
        
        Outputs:
            curve:  a DiscreteCurve forming the shortest path from p1 to p2.

            dist:   a scalar indicating the length of the shortest curve.
        """
        idx1 = self.grid_point(p1)
        idx2 = self.grid_point(p2)
        path = nx.shortest_path(self.G, source=idx1, target=idx2, weight='weight') # list with N elements
        coordinates = self.grid.view(self.grid.shape[0], -1)[:, path] # (dim)xN
        N = len(path)
        curve = DiscreteCurve(begin=coordinates[:, 0], end=coordinates[:, -1], num_nodes=N)
        with torch.no_grad():
            curve.parameters[:, :] = coordinates[:, 1:-1].t()
        dist = 0
        for i in range(N-1):
            dist += self.G.edges[path[i], path[i+1]]['weight']
        return curve, dist

    def connecting_geodesic(self, p1, p2, model, graph_id, curve=None):

        """Compute the shortest path on the discretized manifold and fit
        a smooth curve to the resulting discrete curve.

        Inputs:
            p1:     a torch Tensor corresponding to one latent point.

            p2:     a torch Tensor corresponding to another latent point.
        
        Optional input:
            curve:  a curve that should be fitted to the discrete graph
                    geodesic. By default this is None and a CubicSpline
                    with default paramaters will be constructed.
        
        Outputs:
            curve:  a smooth curve forming the shortest path from p1 to p2.
                    By default the curve is a CubicSpline with its default
                    parameters; this can be changed through the optional
                    curve input.
        """

        dim, xsize, ysize = self.grid.shape

        self.G = copy.deepcopy(self.mem_g)
        # Managing all the obstacles

        device = p1.device
        if len(model.env.via_points) == 0:
            for obstacle in model.env.obstacles:

                point1 = torch.tensor(obstacle.bounding_box[0])
                point2 = torch.tensor(obstacle.bounding_box[1])
                point3 = torch.tensor(obstacle.bounding_box[2])
                point4 = torch.tensor(obstacle.bounding_box[3])

                point1 = self.grid_point(point1)
                point2 = self.grid_point(point2)
                point3 = self.grid_point(point3)
                point4 = self.grid_point(point4)

                start_idx = np.arange(point1, point3 + 1, ysize)
                all_indices = []
                for i in start_idx:
                    all_indices.append(np.arange(i, i + point2 - point1 + 1, 1))
                distances = torch.sqrt(torch.sum(torch.pow(self.decoded_positions - obstacle.position.to_tensor()[:2], 2), 1))

                threshold = 0.02
                sim_vec = torch.nonzero((distances <= threshold))
                for i in np.array(sim_vec).flatten():
                    if self.G.has_edge(i, i - ysize):
                        self.G[i][i - ysize]['weight'] = 1
                    if self.G.has_edge(i, i - 1):
                        self.G[i][i - 1]['weight'] = 1
                    if self.G.has_edge(i, i + 1):
                        self.G[i][i + 1]['weight'] = 1
                    if self.G.has_edge(i, i + ysize):
                        self.G[i][i + ysize]['weight'] = 1

            m = None
            for obstacle_i in model.env.obstacles:
                obstacle_radius = obstacle_i.input_space_radius
                obstacle_position = torch.tensor([obstacle_i.position.x, obstacle_i.position.y, obstacle_i.position.z])
                obs_metric = torch.exp(-torch.sqrt((self.decoded_positions -
                                                    obstacle_position[:2]) ** 2).sum(1).reshape((10, 10))/
                                       (2 * obstacle_radius ** 2))
                if m is not None:
                    m += 1.0 + (100000.0 * obs_metric)
                else:
                    m = 1.0 + (100000.0 * obs_metric)

            idx1 = self.grid_point(p1)
            idx2 = self.grid_point(p2)

            path = nx.shortest_path(self.G, source=idx1, target=idx2, weight='weight') # list with N elements

            weights = [self.G.edges[path[k], path[k+1]]['weight'] for k in range(len(path)-1)]
            self.coordinates = (self.grid.view(self.grid.shape[0], -1)[:, path[1:-1]]).t() # Nx(dim)
            t = torch.tensor(weights[:-1], device=device).cumsum(dim=0) / sum(weights)
        else:
            idx1 = self.grid_point(p1)
            idx2 = self.grid_point(p2)
            path = nx.shortest_path(self.G, source=idx1, target=idx2, weight='weight') # list with N elements
            weights = [self.G.edges[path[k], path[k+1]]['weight'] for k in range(len(path)-1)]
            self.coordinates = (self.grid.view(self.grid.shape[0], -1)[:, path[0:-1]]).t()  # Nx(dim)
            t = torch.tensor(weights, device=device).cumsum(dim=0) / sum(weights)

        if curve is None:
            curve = CubicSpline(p1, p2)
        else:
            curve.begin = p1
            curve.end = p2

        curve.fit(t, self.coordinates)
        # self.draw_graph(graph_id, curve)
        return curve
