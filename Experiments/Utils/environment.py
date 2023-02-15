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


class Environment:
    """Create a simple environment with moving obstacles and via points
    """

    def __init__(self):
        self.obstacles = [] # list of positions
        self.via_points = []

    def add_obstacle(self, obstacle):
        """ Add obstacle to the existing list of obstacles
        Inputs:
            obstacle: a new obstacle instance
        """
        self.obstacles.append(obstacle)

    def add_via_point(self, via_point):
        """Add via-points to the existing list of via_points
        Inputs:
            via_point: a new via-point instance
        """
        self.via_points.append(via_point)

    def step(self):
        """Move the obstacle to its new state at each time step
        """
        for obstacle in self.obstacles:
            obstacle.next_state()