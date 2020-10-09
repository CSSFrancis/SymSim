import numpy as np

from SymSim.utils.rotation_utils import _get_rotation_matrix, _get_random_2d_rot, _get_random_3d_rot
from SymSim.utils.simulation_utils import _get_speckle_size, _get_wavelength, _shape_function, _get_speckle_intensity
from SymSim.utils.vector_utils import rotation_matrix_from_vectors, build_ico
from skimage.draw import circle
from skimage.filters import gaussian
from numpy.random import random, choice
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Cluster(object):
    def __init__(self,
                 symmetry=10,
                 radius=1,
                 k=4.0,
                 position=random(2),
                 rotation_2d=np.eye(3),
                 rotation_3d=np.eye(3),
                 plane_direction=[0,0,1],
                 displacement=None):
        """Defines a sim with a symmetry of symmetry, a radius of radius in nm and position of position.

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
        displacement: float
            The mean squared displacement in nm.
        """
        self.symmetry = symmetry
        self.radius = radius
        self.position = position
        self.rotation_2d = rotation_2d
        self.rotation_3d = rotation_3d
        self.k = k
        self.plane_direction = plane_direction
        self.beam_direction = [0,0,1]
        self.displacement=displacement

    def get_diffraction(self, img_size=8.0,
                        num_pixels=512,
                        accelerating_voltage=200,
                        conv_angle=0.6):
        """Takes some image size in inverse nm and then plots a 'diffraction pattern'
        for the sim.

        Parameters
        ---------
        img_size: float
            The size of the image in nm^-1
        num_pixels: int
            The number of pixels for the image along 1 direction
        accelerating_voltage: int
            The accelerating voltage in KeV
        conv_angle: float
            The convergance angle in mrad
        disorder: float
            The mean squared displacement for the sim.

        """
        sphere_radius = 1/_get_wavelength(accelerating_voltage)
        scale = (num_pixels-1)/img_size
        k_rotated=self.get_k_vectors()
        observed_intensity = [_get_speckle_intensity(k_vector=k,
                                                     ewald_sphere_rad=sphere_radius,
                                                     disorder=self.displacement,
                                                     cluster_rad=self.radius,
                                                     beam_direction=self.beam_direction)
                              for k in k_rotated]
        radius = _get_speckle_size(accelerating_voltage, conv_angle)*scale
        circles = [circle(int(k1[0] * scale + num_pixels/2), int(k1[1] * scale + num_pixels/2),
                          radius=radius) for k1 in k_rotated]
        image = np.ones(shape=(num_pixels,num_pixels))
        for (r, c), i in zip(circles, observed_intensity):
            image[r, c] = i + image[r, c]
        image = gaussian(image=image, sigma=2)
        return image

    def get_k_vectors(self):
        angle = (2 * np.pi) / self.symmetry  # angle between speckles on the pattern
        k = [[np.cos(angle * i) * self.k, np.sin(angle * i) * self.k, 0] for i in
             range(self.symmetry)]  # vectors for the speckles perp to BA
        #print(self.plane_direction)
        if not np.allclose(self.plane_direction, [0, 0, 1]) and not np.allclose(self.plane_direction, [0, 0, -1]) :
            rot = rotation_matrix_from_vectors([0,0,1],self.plane_direction)
            k = [np.dot(rot,v) for v in k]
        k_rotated2d = [np.dot(self.rotation_2d, speckle) for speckle in k]
        k_rotated = [np.dot(self.rotation_3d, speckle) for speckle in k_rotated2d]
        return k_rotated

    def get_speckles(self, img_size=10.0,
                     num_pixels=128,
                     accelerating_voltage=200,
                     conv_angle=0.6):
        """
        This function returns the diffraction speckles as circles as defined by the
        skimage.draw.Circle class. Each speckle also has some intensity associated with it.

        Parameters
        -----------
        img_size:
            The real size of the image that is being projected onto. (In inverse nm)
        num_pixels:
            The pixelated size of the image being projected onto.
        accelerating_voltage:
            The accelerating voltage for getting the Ewald sphere radius
        conv_angle:
            The convergence angle in mrad for beam.
        """
        sphere_radius = 1/_get_wavelength(accelerating_voltage)
        scale = (num_pixels-1)/img_size
        k_rotated = self.get_k_vectors()
        observed_intensity = [_get_speckle_intensity(k_vector=k,
                                                     ewald_sphere_rad=sphere_radius,
                                                     disorder=self.displacement,
                                                     cluster_rad=self.radius,
                                                     beam_direction=self.beam_direction)
                              for k in k_rotated]
        radius = _get_speckle_size(accelerating_voltage, conv_angle)*scale
        speckles = [circle(int(k1[0] * scale + num_pixels/2), int(k1[1] * scale + num_pixels/2),
                           radius=radius, shape=(num_pixels,num_pixels)) for k1 in k_rotated]
        return speckles, observed_intensity

    def get_intensity(self,
                      accelerating_voltage=200):
        """Gives the raw intensity for each speckle

        Parameters
        --------
        accelerating_voltage: int
            The accelerating voltage in keV
        disorder:
            The mean squared displacement for some sim
        """
        sphere_radius = 1 / _get_wavelength(accelerating_voltage)
        k_rotated = self.get_k_vectors()
        observed_intensity = [_get_speckle_intensity(k_vector=k,
                                                     ewald_sphere_rad=sphere_radius,
                                                     disorder=self.displacement,
                                                     cluster_rad=self.radius,
                                                     beam_direction=self.beam_direction)
                              for k in k_rotated]
        return observed_intensity

    def get_angle_between(self):
        return np.arccos((np.trace(self.rotation_3d)-1)/2)


