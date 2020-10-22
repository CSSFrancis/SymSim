import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, choice
from skimage.draw import circle

from SymSim.sim.cluster import Cluster
from SymSim.sim.clusters import get_random_cluster, get_random_icosahedron
from matplotlib.patches import Circle
from hyperspy._signals.signal2d import Signal2D


class SimulationCube(list):
    """Defines a simulation cube of dimensions x, y, z in nm.  This allows you to create some simulation of the cube
    based on kinematic diffraction"""
    def __init__(self,
                 dimensions=(20, 20, 20)):
        """Initializes the simulation cube
        Parameters
        --------------
        dimensions: tuple
            The dimensions of the cube to simulate.
        """
        super().__init__()
        self.dimensions = np.array(dimensions)

    def __str__(self):
        return ("<Cube of " + str(self.num_clusters) + " clusters [" +
                str(self.dimensions[0]) + " x " + str(self.dimensions[1]) +
                "x " + str(self.dimensions[2]) + "nm]>")

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
                                       self.dimensions,
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
        symmetry: list
            The list of symmetries to choose from.  Acceptable symmetries are 2,4,6 and 10
        """
        self.extend(get_random_icosahedron(num_clusters,
                                             self.dimensions,
                                             radius_range,
                                             k_range,
                                             random_rotation))
        return

    def show_projection(self,
                        size=(512,512),
                        acceptance=None):
        """Plots the 2-d projection.  Creates a 2-D projection of the clusters in the amorphous matrix.

        Parameters:
        ------------
        acceptance: float
            The angle of acceptance.  Only clusters within this projection will be allowed.
        size: int
            The size of the image made

        """
        projection = np.zeros(size)
        scale = [self.dimensions[0]/size[0],self.dimensions[1]/size[1]]
        for cluster in self:
            if acceptance is not None:
                if abs(cluster.get_angle_between()) > acceptance:
                    next()
            inten = np.sum(cluster.get_intensity())
            r, c = circle(cluster.position[0]/scale[0],
                          cluster.position[1]/scale[1],
                          radius=cluster.radius/scale[0],
                          shape=size)
            projection[r, c] = inten + projection[r, c]
        return projection

    def plot_symmetries(self,
                        symmetries=[2, 4, 6, 8, 10],
                        norm=1,
                        acceptance=None):
        """Plots the 2-d projection of the symmetries as circles

        Parameters:
        ------------
        acceptance: float
            The angle of acceptance.  Only clusters within this projection will be allowed.
        size: int
            The size of the image made
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0,self.dimensions[0])
        ax.set_ylim(0,self.dimensions[1])
        colors = ["black","blue","red","green","yellow","red","orange", "purple"]
        for cluster in self:
            if acceptance is not None:
                if abs(cluster.get_angle_between()) > acceptance:
                    continue
            inten = np.mean(cluster.get_intensity())
            c = Circle((cluster.position[0],
                        cluster.position[1]),
                        radius=cluster.radius,
                        alpha=inten/norm,
                        color=colors[symmetries.index(cluster.symmetry)])
            ax.add_patch(c)
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker='o', color=colors[i], label=str(sym)+" fold symmetry",
               markerfacecolor=colors[i], markersize=15) for i,sym in enumerate(symmetries)]

        ax.legend(handles=leg)
        return

    def get_4d_stem(self,
                    convergence_angle=.74,
                    accelerating_voltage=200,
                    k_rad = 5.0,
                    simulation_size=(50, 50, 128, 128),
                    noise = False,
                    num_electrons=1000,
                    convolve=False,
                    beam_size=None):
        """Returns an amorphous2d object which shows the 4d STEM projection for some set of clusters along some
        illumination

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
        real_scale = simulation_size[0] / self.dimensions[0]

        for cluster in self:
            speckles, observed_intensity = cluster.get_speckles(img_size=k_rad*2,
                                                                num_pixels=simulation_size[2],
                                                                accelerating_voltage=accelerating_voltage,
                                                                conv_angle=convergence_angle)

            rr, rc = circle(r=cluster.position[0] * real_scale,
                            c=cluster.position[1] * real_scale,
                            radius=cluster.radius * real_scale,
                            shape=(simulation_size[0], simulation_size[1]))
            for (sr,sc), inten in zip(speckles, observed_intensity):
                inner_r, outer_r = np.meshgrid(sr, rr)
                inner_c, outer_c = np.meshgrid(sc, rc)
                if noise:
                    #print(np.size(inner_c))
                    inten = np.random.poisson(inten*num_electrons,size=np.size(inner_c))
                else:
                    inten= inten* num_electrons
                dataset[(outer_c.flatten(),
                         outer_r.flatten(),
                         inner_c.flatten(),
                         inner_r.flatten())] = inten + dataset[(outer_c.flatten(),
                                                                outer_r.flatten(),
                                                                inner_c.flatten(),
                                                                inner_r.flatten())]
        dataset = Signal2D(dataset)
        if convolve:
            if beam_size is None:
                beam_size = .9/convergence_angle  # rough calculation
            from scipy.signal import convolve2d
            num_pixels = int(real_scale*beam_size)
            xx,yy= np.ogrid[-num_pixels-1:num_pixels+2,-num_pixels-1:num_pixels+2]
            kernel =1 - ((xx ** 2 + yy ** 2) ** .5 - num_pixels).clip(0,1)
            dataset = dataset.T
            dataset.map(convolve2d, in2=kernel, mode="same", inplace=True)
            dataset = dataset.T

        dataset.axes_manager.navigation_axes[0].scale = self.dimensions[0]/simulation_size[0]
        dataset.axes_manager.navigation_axes[1].scale = self.dimensions[1] / simulation_size[1]
        dataset.axes_manager.navigation_axes[0].units = "nm"
        dataset.axes_manager.navigation_axes[1].units = "nm"
        dataset.axes_manager.signal_axes[0].scale = k_rad*2 /simulation_size[2]
        dataset.axes_manager.signal_axes[1].scale = k_rad*2 / simulation_size[3]
        dataset.axes_manager.signal_axes[0].units = "$nm^-1$"
        dataset.axes_manager.signal_axes[1].units = "$nm^-1$"
        dataset.axes_manager.signal_axes[0].offset = -k_rad
        dataset.axes_manager.signal_axes[1].offset = -k_rad
        return dataset

