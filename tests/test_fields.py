"""Analytic field identities, not additional electromagnetic simulations."""
import unittest
import numpy as np
from deta_generator3.fixed_fov import periodic_complex_sample
from deta_generator3.vision import hotspots, poynting


class FieldTests(unittest.TestCase):
    def test_complex_interpolation_precedes_intensity_and_rejects_bad_seams(self):
        nodes = np.array([-400., 0., 400.])
        field = np.zeros((3, 3, 1, 3), dtype=np.complex64)
        field[:, :, 0, 0] = [-1, 1, -1]
        samples, check = periodic_complex_sample(field, nodes, nodes, [-200., 200.])
        np.testing.assert_allclose(np.sum(np.abs(samples)**2, -1), 0, atol=1e-12)
        self.assertEqual(check["relative_boundary_max_error"], 0)
        field[:, -1, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "periodic boundaries"):
            periodic_complex_sample(field, nodes, nodes, [-200., 200.])

    def test_fixed_window_does_not_join_hotspots_across_opposite_edges(self):
        image = np.zeros((16, 16))
        image[7:9, :2] = 10
        image[7:9, -2:] = 10
        _, _, finite = hotspots(image, periodic_xy=False)
        _, _, periodic = hotspots(image, periodic_xy=True)
        self.assertEqual((finite["component_count"], periodic["component_count"]), (2, 1))

    def test_poynting_retains_signed_real_power_flow(self):
        electric = np.array([[1+1j, 0, 0]])
        magnetic = np.array([[0, 2+2j, 0]])
        np.testing.assert_allclose(poynting(electric, magnetic), [[0, 0, 2]])
        np.testing.assert_allclose(poynting(electric, -magnetic), [[0, 0, -2]])


if __name__ == "__main__":
    unittest.main()
