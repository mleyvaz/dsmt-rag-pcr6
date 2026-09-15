import math
import unittest

import numpy as np

import run_experiment as ex


class FusionTests(unittest.TestCase):
    def assert_mass(self, mass):
        self.assertTrue(all(v >= -1e-12 for v in mass.values()))
        self.assertTrue(math.isclose(sum(mass.values()), 1.0, rel_tol=0, abs_tol=1e-10))

    def test_dempster_mass_is_normalized(self):
        sources = [ex.simple_support("A", 0.8, "a"), ex.simple_support("C", 0.7, "c")]
        result = ex.dempster_all([s.masses for s in sources], ex.THETA)
        self.assert_mass(result)

    def test_pcr6_mass_is_normalized(self):
        sources = [
            ex.simple_support("A", 0.8, "a"),
            ex.simple_support("B", 0.7, "b"),
            ex.simple_support("C", 0.6, "c"),
        ]
        result = ex.pcr6_all([s.masses for s in sources])
        self.assert_mass(result)

    def test_two_source_pcr6_matches_pcr5_redistribution(self):
        a, c = ex.SEMANTIC["A"], ex.SEMANTIC["C"]
        result = ex.pcr6_all([{a: 1.0}, {c: 1.0}])
        self.assertAlmostEqual(result[a], 0.5)
        self.assertAlmostEqual(result[c], 0.5)

    def test_nonexclusive_intersection_is_retained(self):
        a, b = ex.SEMANTIC["A"], ex.SEMANTIC["B"]
        result = ex.dempster_all([{a: 1.0}, {b: 1.0}], ex.THETA)
        self.assertEqual(result, {frozenset(("A_and_B",)): 1.0})

    def test_coarse_zadeh_case_selects_shared_minor_claim(self):
        s1 = ex.categorical_support({"A": 0.95, "C": 0.05}, "one")
        s2 = ex.categorical_support({"B": 0.94, "C": 0.06}, "two")
        result = ex.dempster_all([ex.coarse_mass(s1), ex.coarse_mass(s2)], ex.COARSE["THETA"])
        self.assertAlmostEqual(result[ex.COARSE["C"]], 1.0)

    def test_duplicate_discount_preserves_simple_support_strength(self):
        source = ex.simple_support("C", 0.75, "same")
        adjusted = ex.provenance_discount([source] * 5)
        combined = ex.dempster_all([s.masses for s in adjusted], ex.THETA)
        self.assertAlmostEqual(combined[ex.SEMANTIC["C"]], 0.75, places=10)
        self.assertAlmostEqual(combined[ex.THETA], 0.25, places=10)

    def test_all_fusion_outputs_are_probabilities(self):
        sources = [ex.simple_support("A", 0.9, "x"), ex.simple_support("C", 0.7, "y")]
        for method in ex.METHODS:
            probs = ex.fuse(sources, method)
            self.assertTrue(np.all(probs >= -1e-12), method)
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=10, msg=method)


if __name__ == "__main__":
    unittest.main()
