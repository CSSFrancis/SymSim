from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from SymSim.sim.simulation_cube import SimulationCube

class TestSimulationCube(TestCase):
    def test_random_init(self):
        cube = SimulationCube()
        cube.add_random_clusters(100)
        print(cube)

    def test_ico_init(self):
        cube = SimulationCube()
        cube.add_icosahedron(1, radius_range=(4., 4.1))
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
        stem = cube.get_4d_stem(noise=True, convolve=True)
        stem.plot()
        plt.show()