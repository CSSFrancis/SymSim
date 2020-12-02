from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from SymSim.sim.cluster import Cluster
from SymSim.sim.simulation_cube import SimulationCube
from SymSim.utils.rotation_utils import _rand_2d_rotation_matrix
from SymSim.sim.EwaldSphere import EwaldSphere


class TestCluster(TestCase):
    def setUp(self):
        self.c = Cluster(symmetry=10, radius=.5, k=4.0, position=(1, 1))

    def test_get_k_vectors(self):
        self.c.rotation_2d = _rand_2d_rotation_matrix()
        print(self.c.rotation_2d)
        k = np.array(self.c.get_k_vectors())
        plt.scatter(k[:,0], k[:,1])
        plt.show()

    def test_get_k_vectors_rot(self):
        #self.c.rotation_3d = np.eye(3)
        self.c.plane_direction=[1,1,1]
        print(self.c.rotation_2d)
        k = np.array(self.c.get_k_vectors())
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(k[:,0],k[:,1],k[:,2])
        self.c.plane_direction = [1, -1, 1]
        print(self.c.rotation_2d)
        k = np.array(self.c.get_k_vectors())
        ax.scatter(k[:, 0], k[:, 1], k[:, 2])
        plt.show()

    def test_get_diffraciton(self):
        diff = self.c.get_diffraction(img_size=15.0)
        plt.imshow(diff)
        plt.show()

    def test_plot_3d(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.c.plot_3d(ax=ax)
        plt.show()

    def test_plot_2d(self):
        self.c.plot_2d()

    def test_plot_2d_sphere(self):
        thick_sphere = EwaldSphere(acc_voltage=200, convergence_angle=0.6)
        self.c.plot_2d(ewald_sphere=thick_sphere)
        plt.show()



class TestSimulationCube(TestCase):
    def test_random_init(self):
        cube = SimulationCube()
        cube.add_random_clusters(100)
        print(cube)

    def test_ico_init(self):
        cube = SimulationCube()
        cube.add_icoso(1, radius_range=(4., 4.1))
        stem = cube.get_4d_stem(noise=True,convolve=True)
        stem.plot()
        plt.show()
        print(cube)

    def test_projection(self):
        cube = SimulationCube()
        cube.add_random_clusters(100)
        projection = cube.show_projection()
        plt.imshow(projection)
        plt.show()

    def test_symmetry_projection(self):
        cube = SimulationCube()
        cube.add_random_clusters(1000)
        projection = cube.plot_symmetries()

    def test_symmetry_projection_acceptance(self):
        cube = SimulationCube()
        cube.add_random_clusters(1000)
        projection = cube.plot_symmetries(acceptance=np.pi/8)
        #projection = cube.plot_symmetries(acceptance=np.pi / 2)

    def test_4dStem(self):
        cube = SimulationCube()
        cube.add_random_clusters(10000)
        #cube.plot_symmetries()
        stem = cube.get_4d_stem(noise=True, disorder=.05, convolve=True)
        stem.plot()
        plt.show()




