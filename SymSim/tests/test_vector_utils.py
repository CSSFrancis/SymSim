import unittest
import numpy
from SymSim.utils.vector_utils import _get_deflection_from_convergence
from SymSim.utils.simulation_utils import _get_wavelength

class TestVectorUtils(unittest.TestCase):

    def test_no_convergence(self):
        radius = 1/ _get_wavelength(200)
        z,xy = _get_deflection_from_convergence(convergence_angle=0, radius=radius)
        numpy.testing.assert_equal(z, 0.0)
        numpy.testing.assert_equal(xy, 0.0)
    def test_convergence(self):
        radius = 1/ _get_wavelength(200)
        dx,dy,dz = _get_deflection_from_convergence(convergence_angle=0.6, radius=radius)
        print(dx,dy,dz)



if __name__ == '__main__':
    unittest.main()
