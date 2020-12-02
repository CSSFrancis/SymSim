from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from SymSim.sim.EwaldSphere import EwaldSphere

class TestEwaldSphere(TestCase):
    def setUp(self):
        self.sphere = EwaldSphere(acc_voltage=200)
        self.thick_sphere = EwaldSphere(acc_voltage=200, convergence_angle=0.6)

    def test_plot_no_covergence(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.sphere.plot()
        plt.show()

    def test_plot_convergence(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.thick_sphere.plot()
        plt.show()

    def test_plot_convergence_2d(self):
        self.thick_sphere.plot_2d()
        plt.show()