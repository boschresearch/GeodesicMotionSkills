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

import numpy as np
from GeodesicMotionGenerator.Experiments.position import Position
import torch


class Obstacle():
    """"
    create a moving obstacle
    """
    def __init__(self, start_position, start_orientation, radius, input_space_radius, shape, step):
        self.position = start_position
        if start_orientation is not None:
            self.orientation = start_orientation
        self.radius = radius
        self.input_space_radius = input_space_radius
        self.shape = shape
        self.step = step
        self.bounding_box = self.find_bounding_box()
        self.latent_pos = None

    def find_bounding_box(self):
        """
        compute the bounding box of the obstacle (depreciated)

        output:     list of bounding box positions of the obstacle in the latent space
        """
        point_1 = [self.position.x - self.radius, self.position.y - self.radius]
        point_2 = [self.position.x - self.radius, self.position.y + self.radius]
        point_3 = [self.position.x + self.radius, self.position.y - self.radius]
        point_4 = [self.position.x + self.radius, self.position.y + self.radius]
        return [point_1, point_2, point_3, point_4]

    def next_state(self):
        """"
        compute the next state of the obstacle in the latent space (depreciated)
        """
        self.position += self.step

    def decode_position(self, model):
        """
         compute the ambient space position of the obstacle (depreciated)
        """
        return model.decode(torch.tensor([self.position.x, self.position.y]))[0].mean