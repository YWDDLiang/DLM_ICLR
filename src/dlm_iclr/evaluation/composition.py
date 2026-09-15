"""Opt-in SMACT 3 composition validity with a mixed-valence supplement.

The mixed-element scope and per-atom expansion follow SMACT v4.0.0:
https://github.com/WMD-group/SMACT/blob/v4.0.0/smact/screening.py
The oxidation data, Pauling test and original acceptance remain SMACT 3.1.
Equivalent permutations of atoms of one element are enumerated only once.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement, product
import math

import smact
from pymatgen.core import Composition
from smact.screening import pauling_test

from dlm_iclr._vendor.crysllmgen.validity import smact_validity


MIXED_ELEMENTS = frozenset("Fe Mn Co Cu Ni V Ti Cr Nb Mo W Re Ru Os Pd Ag Au Sn Sb Bi Ce Eu Yb U".split())


def mixed_assignment(symbols, counts, oxidation_states, electronegativities, mixed_elements=MIXED_ELEMENTS):
    """Find one charge-neutral, Pauling-valid assignment or exhaust the support.

    counts are reduced-formula counts, as in the upstream screening function.
    Only elements in mixed_elements expand into independently charged atoms.
    Each option stores total charge and distinct oxidation states; charge and
    Pauling predicates are invariant under permutations within one element.
    """
    options = []
    for symbol, count, states in zip(symbols, counts, oxidation_states, strict=True):
        states = sorted(set(states or ()))
        if not states:
            return None
        if symbol in mixed_elements:
            assignments = combinations_with_replacement(states, count)
        else:
            assignments = (tuple([state] * count) for state in states)
        options.append([(sum(values), values) for values in assignments])
    if not options:
        return None
    # Exact charge lookup avoids evaluating the final element at every tuple.
    last = {}
    for charge, values in options[-1]:
        last.setdefault(charge, []).append(values)
    for prefix in product(*options[:-1]):
        charge = sum(option[0] for option in prefix)
        for tail in last.get(-charge, ()):
            assignments = [option[1] for option in prefix] + [tail]
            expanded_states, expanded_enegs = [], []
            for values, eneg in zip(assignments, electronegativities, strict=True):
                unique_states = sorted(set(values))
                expanded_states.extend(unique_states)
                expanded_enegs.extend([eneg] * len(unique_states))
            try:
                valid = pauling_test(expanded_states, expanded_enegs)
            except TypeError:
                valid = True  # Preserve the original missing-EN convention.
            if valid:
                return {
                    symbol: {str(o): n for o, n in Counter(values).items()}
                    for symbol, values in zip(symbols, assignments, strict=True)
                }
    return None


@lru_cache(maxsize=8192)
def composition_validity(formula):
    """Return the standard result, supplemented result and a witness when added."""
    composition = Composition(formula)
    elements = sorted(composition.elements, key=lambda element: element.Z)
    amounts = [composition[element] for element in elements]
    if not amounts or any(value <= 0 or value != int(value) for value in amounts):
        raise ValueError("Composition audit requires positive integer atom counts")
    divisor = math.gcd(*(int(value) for value in amounts))
    counts = tuple(int(value) // divisor for value in amounts)
    symbols = tuple(element.symbol for element in elements)
    standard = bool(smact_validity(tuple(element.Z for element in elements), counts))
    base = dict(
        reduced_formula=composition.reduced_formula,
        standard=standard,
        mixed_supplement=False,
        comp_valid=standard,
        witness=None,
    )
    if standard:
        return dict(base, reason="smact3_standard")
    space = smact.element_dictionary(symbols)
    states = [space[symbol].oxidation_states or [] for symbol in symbols]
    if any(not values for values in states):
        return dict(base, reason="oxidation_state_missing")
    if not any(symbol in MIXED_ELEMENTS and count > 1 for symbol, count in zip(symbols, counts)):
        return dict(base, reason="no_expandable_mixed_element")
    witness = mixed_assignment(symbols, counts, states, [space[symbol].pauling_eneg for symbol in symbols])
    if witness is None:
        return dict(base, reason="mixed_charge_or_pauling_failed")
    return dict(
        base, comp_valid=True, mixed_supplement=True, witness=witness, reason="mixed_valence_supplement"
    )
