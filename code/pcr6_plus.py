"""Exact multi-source PCR6 and PCR6+ on finite Shafer frames.

Focal elements are represented as ``frozenset`` objects.  The PCR6+
implementation follows Dezert and Smarandache (2021), equations (23) and
(26): binary keeping indexes are computed from the distinct focal elements
of each conflicting tuple, and conflict is redistributed only to elements
whose index equals one.

The implementation deliberately enumerates focal tuples.  It is intended as
a transparent reference implementation for experiments, not as a
large-frame optimizer.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


Focal = frozenset[str]
Mass = dict[Focal, float]


class TotalConflictError(ValueError):
    """Raised when Dempster normalization is undefined because K = 1."""


def normalize_mass(mass: Mapping[Focal, float], *, atol: float = 1e-14) -> Mass:
    """Remove numerical zeros and normalize a nonnegative mass mapping."""
    cleaned = {frozenset(k): float(v) for k, v in mass.items() if float(v) > atol}
    if any(v < -atol for v in mass.values()):
        raise ValueError("A BBA cannot contain negative mass")
    total = math.fsum(cleaned.values())
    if total <= atol:
        raise ValueError("A BBA must have positive total mass")
    return {k: v / total for k, v in cleaned.items()}


def validate_bbas(masses: Sequence[Mapping[Focal, float]]) -> list[Mass]:
    if len(masses) < 2:
        raise ValueError("At least two BBAs are required")
    normalized = [normalize_mass(m) for m in masses]
    if any(frozenset() in m for m in normalized):
        raise ValueError("Closed-world input BBAs cannot assign mass to the empty set")
    return normalized


def focal_tuples(masses: Sequence[Mapping[Focal, float]]):
    """Yield focal tuples, their source masses, product, and intersection."""
    for combo in itertools.product(*(list(m.items()) for m in masses)):
        focals = tuple(item[0] for item in combo)
        values = tuple(float(item[1]) for item in combo)
        product = math.prod(values)
        intersection = frozenset.intersection(*focals)
        yield focals, values, product, intersection


def conjunctive_all(masses: Sequence[Mapping[Focal, float]]) -> Mass:
    """Unnormalized exact multi-source conjunctive combination."""
    bbas = validate_bbas(masses)
    out: defaultdict[Focal, float] = defaultdict(float)
    for _, _, product, intersection in focal_tuples(bbas):
        out[intersection] += product
    return dict(out)


def dempster_all(masses: Sequence[Mapping[Focal, float]], *, atol: float = 1e-12) -> Mass:
    """Exact multi-source Dempster rule; raise when total conflict makes it undefined."""
    conjunctive = conjunctive_all(masses)
    conflict = conjunctive.pop(frozenset(), 0.0)
    if conflict >= 1.0 - atol:
        raise TotalConflictError("Dempster's rule is undefined at total conflict (K = 1)")
    return normalize_mass({focal: value / (1.0 - conflict) for focal, value in conjunctive.items()})


def pcr6_all(masses: Sequence[Mapping[Focal, float]]) -> Mass:
    """Exact simultaneous multi-source PCR6 (Dezert-Smarandache formulation)."""
    bbas = validate_bbas(masses)
    out: defaultdict[Focal, float] = defaultdict(float)
    for focals, values, product, intersection in focal_tuples(bbas):
        if intersection:
            out[intersection] += product
            continue
        denominator = math.fsum(values)
        if denominator <= 0.0:
            continue
        for focal, value in zip(focals, values):
            out[focal] += product * value / denominator
    return normalize_mass(out)


def binary_keeping_indices(focals: Iterable[Focal]) -> dict[Focal, int]:
    """Compute PCR6+ binary keeping indexes using Eq. (23).

    Only distinct focal elements are considered.  For candidate ``xi``, the
    product in Eq. (23) ranges over every ordered pair of distinct elements
    ``(small, large)`` satisfying ``|xi| <= |large|`` and
    ``|small| <= |large|``.  The index is zero exactly when all corresponding
    containment indicators equal one.
    """
    distinct = tuple(dict.fromkeys(focals))
    if not distinct:
        return {}
    result: dict[Focal, int] = {}
    for xi in distinct:
        all_contained = True
        pair_seen = False
        for small in distinct:
            for large in distinct:
                if small == large:
                    continue
                if len(xi) <= len(large) and len(small) <= len(large):
                    pair_seen = True
                    if not small.issubset(large):
                        all_contained = False
                        break
            if not all_contained:
                break
        # A conflicting tuple always has at least two distinct focal elements.
        result[xi] = 0 if pair_seen and all_contained else 1
    return result


def pcr6_plus_all(masses: Sequence[Mapping[Focal, float]]) -> Mass:
    """Exact simultaneous multi-source PCR6+ (Eq. 26)."""
    bbas = validate_bbas(masses)
    out: defaultdict[Focal, float] = defaultdict(float)
    for focals, values, product, intersection in focal_tuples(bbas):
        if intersection:
            out[intersection] += product
            continue

        keeping = binary_keeping_indices(focals)
        weights: defaultdict[Focal, float] = defaultdict(float)
        for focal, value in zip(focals, values):
            if keeping[focal] == 1:
                weights[focal] += value
        denominator = math.fsum(weights.values())
        if denominator <= 0.0:
            raise ArithmeticError("PCR6+ found a conflicting tuple with no kept focal element")
        for focal, weight in weights.items():
            out[focal] += product * weight / denominator
    return normalize_mass(out)


def pignistic(mass: Mapping[Focal, float], atoms: Sequence[str]) -> list[float]:
    """Return BetP probabilities for an ordered list of singleton atoms."""
    scores = {atom: 0.0 for atom in atoms}
    for focal, value in mass.items():
        if not focal:
            continue
        share = float(value) / len(focal)
        for atom in focal:
            if atom in scores:
                scores[atom] += share
    total = math.fsum(scores.values())
    if total <= 0.0:
        return [1.0 / len(atoms)] * len(atoms)
    return [scores[atom] / total for atom in atoms]


def dsmp(mass: Mapping[Focal, float], atoms: Sequence[str], *, epsilon: float = 0.001) -> list[float]:
    """Return the DSmP-epsilon subjective probability on a Shafer frame.

    Compound mass is redistributed among its singleton members in proportion
    to their singleton masses plus ``epsilon``.  The default follows the small
    positive value recommended by Dezert and Smarandache for degenerate cases.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")
    singleton_mass = {atom: float(mass.get(frozenset((atom,)), 0.0)) for atom in atoms}
    scores = {atom: 0.0 for atom in atoms}
    for focal, value in mass.items():
        members = [atom for atom in atoms if atom in focal]
        if not members or float(value) == 0.0:
            continue
        denominator = math.fsum(singleton_mass[atom] + epsilon for atom in members)
        for atom in members:
            scores[atom] += float(value) * (singleton_mass[atom] + epsilon) / denominator
    total = math.fsum(scores.values())
    if total <= 0.0:
        return [1.0 / len(atoms)] * len(atoms)
    return [scores[atom] / total for atom in atoms]


def singleton_belief_scores(mass: Mapping[Focal, float], atoms: Sequence[str]) -> list[float]:
    """Singleton belief scores for the maximum-belief decision rule."""
    return [float(mass.get(frozenset((atom,)), 0.0)) for atom in atoms]


def singleton_plausibility_scores(mass: Mapping[Focal, float], atoms: Sequence[str]) -> list[float]:
    """Singleton plausibility scores for maximum-plausibility decisions."""
    return [math.fsum(float(value) for focal, value in mass.items() if atom in focal) for atom in atoms]


def belief_interval_distance_scores(mass: Mapping[Focal, float], atoms: Sequence[str]) -> list[float]:
    """Scores induced by minimum Euclidean belief-interval distance.

    This implements the decision criterion of Dezert et al. (Belief 2016)
    using the Euclidean-family belief-interval distance of Han, Dezert, and
    Yang (IEEE TSMC: Systems, 2018).  For each singleton decision ``atom``,
    the BBA is compared with the categorical BBA focused on that singleton.
    The returned score is ``1 - d_BI`` so that larger values are preferred.

    The exact distance enumerates every nonempty subset of the finite frame.
    It is therefore intended for the small answer frames used in this study.
    """
    atoms = tuple(atoms)
    if not atoms:
        raise ValueError("At least one atom is required")
    frame = frozenset(atoms)
    normalized = normalize_mass(mass)
    if any(not focal.issubset(frame) for focal in normalized):
        raise ValueError("Every focal element must be a subset of the supplied frame")

    subsets = [
        frozenset(combo)
        for cardinality in range(1, len(atoms) + 1)
        for combo in itertools.combinations(atoms, cardinality)
    ]

    intervals = []
    for subset in subsets:
        belief = math.fsum(value for focal, value in normalized.items() if focal.issubset(subset))
        plausibility = math.fsum(value for focal, value in normalized.items() if focal & subset)
        intervals.append((subset, belief, plausibility))

    normalization = 1.0 / (2 ** (len(atoms) - 1))
    scores = []
    for atom in atoms:
        squared = 0.0
        for subset, belief, plausibility in intervals:
            categorical = 1.0 if atom in subset else 0.0
            midpoint_delta = (belief + plausibility) / 2.0 - categorical
            radius_delta = (plausibility - belief) / 2.0
            squared += midpoint_delta**2 + radius_delta**2 / 3.0
        distance = math.sqrt(normalization * squared)
        scores.append(1.0 - distance)
    return scores


def nonspecificity(mass: Mapping[Focal, float]) -> float:
    """Hartley nonspecificity, sum m(A) log2 |A|."""
    return math.fsum(float(value) * math.log2(len(focal)) for focal, value in mass.items() if focal)
