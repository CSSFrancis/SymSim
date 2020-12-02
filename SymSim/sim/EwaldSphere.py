import numpy as np

from SymSim.utils.rotation_utils import _get_rotation_matrix, _get_random_2d_rot, _get_random_3d_rot
from SymSim.utils.simulation_utils import _get_speckle_size, _get_wavelength, _shape_function, _get_speckle_intensity
from SymSim.utils.vector_utils import rotation_matrix_from_vectors, \
    build_ico, _get_distance_from_sphere,\
    _get_deflection_from_convergence, _get_distance_from_sphere_delta
from skimage.draw import circle
from skimage.filters import gaussian
from numpy.random import random, choice
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d


class EwaldSphere():

    def __init__(self, acc_voltage, convergence_angle=None):
        """
        Parameters
        ----------
        wavelength: float
            The wavelength in nm
        :param wavelength:
        :param convergence_angle:
        """
        self.wavelength = _get_wavelength(acc_voltage)
        self.acc_voltage = acc_voltage
        self.convergence_angle = convergence_angle
        self.radius = 1/self.wavelength

    def plot_2d(self, ax=None):
        if ax is None:
            ax = plt
        if self.convergence_angle is None:
            x, = np.mgrid[-10:10:.1]
            z = _get_distance_from_sphere(x, 0,radius=self.radius)
            ax.plot(x,z, color="black", label="Ewald Sphere")
            ax.plot([0,0],[.5,-0.1], color="black", alpha=0.5, label="Beam Direction")
            ax.legend()
            ax.xlabel("$k_x$, $nm^{-1}$")
            ax.ylabel("$k_z$, $nm^{-1}$")
        else:
            x= np.mgrid[-10:10:.1]
            z = _get_distance_from_sphere(x, 0,radius=self.radius)
            dx, dy, dz = _get_deflection_from_convergence(convergence_angle=self.convergence_angle,
                                                          radius=self.radius)
            min_z = _get_distance_from_sphere_delta(x,0,radius=self.radius, deltaxy=dx, deltaz=dz)
            max_z = _get_distance_from_sphere_delta(x,0,radius=self.radius, deltaxy=-dx, deltaz=dz)
            ax.plot(x, min_z,color="red",label="Ewald Sphere Edge")
            ax.plot(x, max_z, color="red")
            ax.plot(x,z, color="black", alpha=0.5)
            ax.plot([0,0],[.5,-0.1], color="black", alpha=0.5, label="Beam Direction")
            ax.legend()
            #ax.xlabel("$k_x$, $nm^{-1}$")
            #ax.ylabel("$k_z$, $nm^{-1}$")

    def plot(self, ax):
        if self.convergence_angle is None:
            x,y = np.mgrid[-10:10:.1,-10:10:.1]
            x, y = [x,y]
            z = _get_distance_from_sphere(x, y,radius=self.radius)
            print("This is z: ",z)
            ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
            ax.set_title('Surface plot')

        else:
            x,y = np.mgrid[-10:10:.1,-10:10:.1]
            x, y = [x,y]
            z = _get_distance_from_sphere(x, y,radius=self.radius)
            dx, dy, dz = _get_deflection_from_convergence(convergence_angle=self.convergence_angle,
                                                          radius=self.radius)
            min_z = _get_distance_from_sphere_delta(x,y,radius=self.radius, deltaxy=dx, deltaz=dz)
            max_z = _get_distance_from_sphere_delta(x,y,radius=self.radius, deltaxy=-dx, deltaz=dz)
            print("This is z: ",z)
            ax.plot_surface(x, y, min_z, cmap='Reds', edgecolor='none',alpha=0.5)
            print("this is minz", min_z)
            ax.plot_surface(x, y,max_z, cmap='Blues', edgecolor='none', alpha=.5)
            ax.set_title('Surface plot')
