import numpy as np

from SymSim.utils.rotation_utils import _get_rotation_matrix, _get_random_2d_rot, _get_random_3d_rot
from SymSim.utils.simulation_utils import _get_speckle_size, _get_wavelength, _shape_function, _get_speckle_intensity
from SymSim.utils.vector_utils import rotation_matrix_from_vectors, build_ico
from skimage.draw import circle
from skimage.filters import gaussian
from numpy.random import random, choice
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class ShapeFunction(object):
    def __init__(self, function):
        """The shape function of the sim.  This is a generic constructor class for visualizing
        and refining the shape function.

        Parameters:
        ----------------
        symmetry: int
            The symmetry of the sim being simulated
        radius: float
            The radius of the sim in nm
        position: tuple
            The position of the sim in the simulation cube
        rotation_vector: tuple
            The vector which the sim is rotated around
        rotation_angle: float
            The angle the sim is rotated about
        """
        self.symmetry = symmetry
        self.radius = radius
        self.position = position
        self.rotation_2d = rotation_2d
