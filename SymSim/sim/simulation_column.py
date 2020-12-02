import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, choice
from skimage.draw import circle

from SymSim.sim.cluster import Cluster
from SymSim.sim.clusters import get_random_cluster, get_random_icosahedron
from matplotlib.patches import Circle
from hyperspy._signals.signal2d import Signal2D


class SimulationColumn(list):
    """Defines a simulation column. This is just a set of clusters all in one column.  The purpose of this is to
    see how multiple columns of some of dimensions x, y, z in nm.  This allows you to create some simulation of the cube
    based on kinematic diffraction"""
    def __init__(self):
        """Initializes the simulation column
        """
        super().__init__()

    def __str__(self):
        return "<Column of " + str(self.num_clusters) + " clusters"

    @property
    def num_clusters(self):
        return len(self)

    def add_random_clusters(self, num_clusters,
                            radius_range=(.5, 1.0),
                            k_range=(3.5, 4.5),
                            random_rotation=True,
                            symmetry=[2, 4, 6, 10]):
        """Randomly initializes the glass with a set of random clusters.
        Parameters
        ------------
        num_clusters: int
            The number of cluster to add
        radius_range: tuple
            The range of radii to randomly choose from
        k_range: tuple
            THe range of k to randomly choose from
        random_rotation: bool
            Random rotate the cluster or not
        symmetry: list
            The list of symmetries to choose from.  Acceptable symmetries are 2,4,6 and 10
        """
        self.extend(get_random_cluster(num_clusters,
                                       (1, 1, 1),
                                       radius_range,
                                       k_range,
                                       random_rotation,
                                       symmetry))
        return

    def add_icosahedron(self,
                        num_clusters,
                        radius_range=(.5, 1.0),
                        k_range=(3.5, 4.5),
                        random_rotation=True):
        """This adds in the appropriate planar symmetries for an icosahedron
        Parameters
        ------------
        num_clusters: int
            The number of cluster to add
        radius_range: tuple
            The range of radii to randomly choose from
        k_range: tuple
            THe range of k to randomly choose from
        random_rotation: bool
            Random rotate the cluster or not
        """
        self.extend(get_random_icosahedron(num_clusters,
                                          (1,1,1),
                                          radius_range,
                                          k_range,
                                          random_rotation))
        return

    def get_diffraction(self,
                        convergence_angle=.74,
                        accelerating_voltage=200,
                        k_rad = 5.0,
                        simulation_size=(256, 256),
                        noise = False,
                        num_electrons=1000,
                        ):
        """Returns an amorphous2d Diffraction pattern based on the clusters in one column.

        Parameters
        ------------
        convergence_angle: float
            The convergence angle for the experiment
        accelerating_voltage: float
            The accelerating voltage for the experiment in kV
        simulation_size: tuple
            The size of the image for both the reciporical space image and the real space image.


        Returns
        ------------
        dataset: Amorphus2D
            Returns a 4 dimensional dataset which represents the cube
        """
        # dataset = Signal2D(np.ones(simulation_size))
        dataset = np.ones(simulation_size)
        for cluster in self:
            speckles, observed_intensity = cluster.get_speckles(img_size=k_rad*2,
                                                                num_pixels=simulation_size[0],
                                                                accelerating_voltage=accelerating_voltage,
                                                                conv_angle=convergence_angle)
            print(cluster)
            for (sr,sc), inten in zip(speckles,observed_intensity):
                dataset[sr, sc] = dataset[sr, sc] + inten
        if noise:
            dataset = dataset*num_electrons
            noise = np.random.poisson(dataset)
            dataset = dataset+noise
        else:
            dataset = dataset * num_electrons
        dataset = Signal2D(dataset)
        dataset.axes_manager.signal_axes[0].scale = k_rad*2 / simulation_size[0]
        dataset.axes_manager.signal_axes[1].scale = k_rad*2 / simulation_size[1]
        dataset.axes_manager.signal_axes[0].units = "$nm^-1$"
        dataset.axes_manager.signal_axes[1].units = "$nm^-1$"
        dataset.axes_manager.signal_axes[0].offset = -k_rad
        dataset.axes_manager.signal_axes[1].offset = -k_rad
        return dataset