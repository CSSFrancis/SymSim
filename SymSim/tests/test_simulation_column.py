from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from SymSim.sim.simulation_column import SimulationColumn
from SymSim.sim.cluster import Cluster

class TestSimulationColumn(TestCase):
    def test_init(self):
        col = SimulationColumn()
        print(col)

    def test_ico_init(self):
        col = SimulationColumn()
        col.add_icosahedron(1, radius_range=(4., 4.1))
        stem = col.get_diffraction(noise=True)
        stem.plot()
        plt.show()

    def test_get_diffraction(self):
        col = SimulationColumn()
        for i in range(2, 11, 2):
            col.append(Cluster(symmetry=i))
        stem = col.get_diffraction(noise=True)
        stem = col.get_diffraction(noise=False)

