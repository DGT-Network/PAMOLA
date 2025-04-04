# import unittest
# import numpy as np
#
# from pamola.pamola_core.metrics.fidelity.statistical_fidelity import kullback_leibler_divergence
#
# class TestFullbackLibelerDivergence(unittest.TestCase):
#
#     def test_identical_distributions(self):
#         p = np.array([0.5, 0.5])
#         q = np.array([0.5, 0.5])
#         self.assertAlmostEqual(kullback_leibler_divergence(p, q), 0.0)
#
#     def test_different_distributions(self):
#         p = np.array([0.9, 0.1])
#         q = np.array([0.5, 0.5])
#         kl_divergence = kullback_leibler_divergence(p, q)
#         self.assertGreater(kl_divergence, 0.0)
#
#     def test_zero_in_q(self):
#         p = np.array([1.0, 0.0, 0.0])
#         q = np.array([0.0, 1.0, 0.0])
#         kl_divergence = kullback_leibler_divergence(p, q)
#         self.assertTrue(np.isinf(kl_divergence) or np.isnan(kl_divergence))
#
#     def test_normalization(self):
#         p = np.array([1, 2, 3])
#         q = np.array([0.1, 0.2, 0.7])
#         kl_divergence = kullback_leibler_divergence(p, q)
#         self.assertIsInstance(kl_divergence, float)
#
#     def test_base_parameter(self):
#         p = np.array([0.5, 0.5])
#         q = np.array([0.5, 0.5])
#         kl_divergence_natural = kullback_leibler_divergence(p, q)
#         kl_divergence_log2 = kullback_leibler_divergence(p, q, base=2)
#         self.assertAlmostEqual(kl_divergence_natural, kl_divergence_log2 * np.log(2))
#
# if __name__ == '__main__':
#     unittest.main()