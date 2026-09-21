import math
import unittest

from pcr6_plus import (
    belief_interval_distance_scores,
    binary_keeping_indices,
    dempster_all,
    dsmp,
    pcr6_all,
    pcr6_plus_all,
    pignistic,
    singleton_belief_scores,
    singleton_plausibility_scores,
)


class PCR6PlusTests(unittest.TestCase):
    def assert_bba(self, mass):
        self.assertTrue(all(value >= -1e-12 for value in mass.values()))
        self.assertAlmostEqual(math.fsum(mass.values()), 1.0, places=12)

    def test_dsmp_reproduces_published_two_atom_example(self):
        a = frozenset(("A",))
        b = frozenset(("B",))
        ab = frozenset(("A", "B"))
        probabilities = dsmp({a: 0.3, b: 0.1, ab: 0.6}, ("A", "B"))
        self.assertAlmostEqual(probabilities[0], 0.7492537313432835, places=12)
        self.assertAlmostEqual(probabilities[1], 0.2507462686567164, places=12)

    def test_decision_transforms_are_well_formed(self):
        a = frozenset(("A",))
        b = frozenset(("B",))
        ab = frozenset(("A", "B"))
        mass = {a: 0.2, b: 0.1, ab: 0.7}
        self.assertAlmostEqual(sum(pignistic(mass, ("A", "B"))), 1.0)
        self.assertAlmostEqual(sum(dsmp(mass, ("A", "B"))), 1.0)
        self.assertEqual(singleton_belief_scores(mass, ("A", "B")), [0.2, 0.1])
        plausibility = singleton_plausibility_scores(mass, ("A", "B"))
        self.assertAlmostEqual(plausibility[0], 0.9)
        self.assertAlmostEqual(plausibility[1], 0.8)

    def test_belief_interval_distance_prefers_categorical_focus(self):
        a = frozenset(("A",))
        b = frozenset(("B",))
        self.assertEqual(belief_interval_distance_scores({a: 1.0}, ("A", "B")), [1.0, 0.0])
        self.assertEqual(belief_interval_distance_scores({b: 1.0}, ("A", "B")), [0.0, 1.0])

    def test_belief_interval_distance_uses_full_belief_interval(self):
        a = frozenset(("A",))
        b = frozenset(("B",))
        ab = frozenset(("A", "B"))
        scores = belief_interval_distance_scores({a: 0.6, b: 0.1, ab: 0.3}, ("A", "B"))
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_published_example_2(self):
        """Reproduce Dezert-Smarandache (2021), Example 2 revisited."""
        a, b, theta = frozenset("A"), frozenset("B"), frozenset(("A", "B"))
        masses = [
            {a: 0.6, b: 0.1, theta: 0.3},
            {a: 0.5, b: 0.3, theta: 0.2},
            {a: 0.4, b: 0.1, theta: 0.5},
        ]
        ordinary = pcr6_all(masses)
        improved = pcr6_plus_all(masses)
        self.assertAlmostEqual(ordinary[a], 0.743496, places=6)
        self.assertAlmostEqual(ordinary[b], 0.162245, places=6)
        self.assertAlmostEqual(ordinary[theta], 0.094259, places=6)
        self.assertAlmostEqual(improved[a], 0.788847, places=6)
        self.assertAlmostEqual(improved[b], 0.181153, places=6)
        self.assertAlmostEqual(improved[theta], 0.03, places=12)

    def test_published_example_6_keeping_indexes(self):
        a = frozenset("A")
        bc = frozenset(("B", "C"))
        ac = frozenset(("A", "C"))
        abc = frozenset(("A", "B", "C"))
        theta = frozenset(("A", "B", "C", "D"))
        indexes = binary_keeping_indices((a, bc, ac, bc, abc, theta))
        self.assertEqual(indexes, {a: 1, bc: 1, ac: 1, abc: 0, theta: 0})

    def test_two_source_pcr6_plus_equals_pcr6(self):
        a, b, theta = frozenset("A"), frozenset("B"), frozenset(("A", "B"))
        masses = [{a: 0.7, theta: 0.3}, {b: 0.8, theta: 0.2}]
        ordinary = pcr6_all(masses)
        improved = pcr6_plus_all(masses)
        self.assertEqual(set(ordinary), set(improved))
        for focal in ordinary:
            self.assertAlmostEqual(ordinary[focal], improved[focal], places=12)

    def test_vacuous_bba_is_neutral_for_pcr6_plus(self):
        a, b, theta = frozenset("A"), frozenset("B"), frozenset(("A", "B"))
        base = [{a: 0.6, b: 0.1, theta: 0.3}, {a: 0.5, b: 0.3, theta: 0.2}, {a: 0.4, b: 0.1, theta: 0.5}]
        before = pcr6_plus_all(base)
        after = pcr6_plus_all(base + [{theta: 1.0}])
        self.assertEqual(set(before), set(after))
        for focal in before:
            self.assertAlmostEqual(before[focal], after[focal], places=12)

    def test_dempster_is_neutral_to_vacuous_bba(self):
        a, b, theta = frozenset("A"), frozenset("B"), frozenset(("A", "B"))
        base = [{a: 0.7, theta: 0.3}, {b: 0.6, theta: 0.4}]
        before = dempster_all(base)
        after = dempster_all(base + [{theta: 1.0}])
        for focal in before:
            self.assertAlmostEqual(before[focal], after[focal], places=12)

    def test_outputs_are_normalized_probabilities(self):
        a, b, c, ab, theta = (
            frozenset("A"),
            frozenset("B"),
            frozenset("C"),
            frozenset(("A", "B")),
            frozenset(("A", "B", "C")),
        )
        masses = [{a: 0.55, ab: 0.30, theta: 0.15}, {b: 0.60, ab: 0.20, theta: 0.20}, {c: 0.45, ab: 0.35, theta: 0.20}]
        for rule in (dempster_all, pcr6_all, pcr6_plus_all):
            fused = rule(masses)
            self.assert_bba(fused)
            probs = pignistic(fused, ("A", "B", "C"))
            self.assertTrue(all(value >= 0.0 for value in probs))
            self.assertAlmostEqual(sum(probs), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
