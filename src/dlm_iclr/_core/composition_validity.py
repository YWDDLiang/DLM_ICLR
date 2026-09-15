from __future__ import annotations
from collections import Counter
from functools import reduce
import itertools
import math
from typing import Any, Dict, List, Sequence, Tuple
from dlm_iclr._core.fixed_slot import CHEMICAL_SYMBOLS


def reduced_composition(atom_types: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return sorted element atomic numbers and gcd-reduced counts."""
    counter = Counter((int(value) for value in atom_types))
    items = [(elem, counter[elem]) for elem in sorted(counter)]
    if not items:
        return ((), ())
    elems, counts = zip(*items)
    gcd_value = reduce(math.gcd, (int(count) for count in counts))
    if gcd_value <= 0:
        gcd_value = 1
    return (tuple((int(elem) for elem in elems)), tuple((int(count // gcd_value) for count in counts)))


def element_symbols(elems: Sequence[int]) -> Tuple[str, ...]:
    """Map atomic numbers to symbols with CrysLLMGen's invalid-value fallback."""
    symbols: List[str] = []
    for elem in elems:
        atomic_number = int(elem)
        if atomic_number <= 0 or atomic_number > 119:
            atomic_number = 1
        if atomic_number < len(CHEMICAL_SYMBOLS):
            symbols.append(CHEMICAL_SYMBOLS[atomic_number])
        else:
            symbols.append("H")
    return tuple(symbols)


def _neutral_ratios_compat(
    smact_module: Any, ox_states: Sequence[int], stoichs: Sequence[Tuple[int, ...]], threshold: int
) -> Tuple[bool, Sequence[Any]]:
    """Handle SMACT 3.x and 4.x neutral_ratios return shapes."""
    result = smact_module.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
    if isinstance(result, tuple) and len(result) == 2:
        cn_e, cn_r = result
        return (bool(cn_e), cn_r)
    if result is None:
        return (False, [])
    if isinstance(result, list):
        return (bool(result), result)
    return (False, [])


def classify_smact_validity(
    elems: Sequence[int], counts: Sequence[int], *, use_pauling_test: bool = True, include_alloys: bool = True
) -> Dict[str, Any]:
    """Approximate why a composition passes or fails the CrysLLMGen SMACT check."""
    import numpy as np
    import smact
    from smact.screening import pauling_test

    elem_symbols = element_symbols(elems)
    if not elem_symbols:
        return {"valid": False, "reason": "empty_composition"}
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [element for _, element in space.items()]
    electronegs = [element.pauling_eneg for element in smact_elems]
    ox_combos = [element.oxidation_states or [] for element in smact_elems]
    if len(set(elem_symbols)) == 1:
        return {"valid": True, "reason": "single_element_shortcut"}
    if include_alloys:
        is_metal_list = [elem_symbol in smact.metals for elem_symbol in elem_symbols]
        if all(is_metal_list):
            return {"valid": True, "reason": "all_metal_shortcut"}
    if any((len(combo) == 0 for combo in ox_combos)):
        return {"valid": False, "reason": "oxidation_state_missing"}
    threshold = np.max(counts)
    charge_neutral_seen = False
    pauling_error_seen = False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(count,) for count in counts]
        cn_e, cn_r = _neutral_ratios_compat(smact, ox_states, stoichs, threshold)
        if not cn_e:
            continue
        charge_neutral_seen = True
        if use_pauling_test:
            try:
                electroneg_ok = pauling_test(ox_states, electronegs)
            except TypeError:
                electroneg_ok = True
                pauling_error_seen = True
        else:
            electroneg_ok = True
        if electroneg_ok and cn_r:
            return {
                "valid": True,
                "reason": "charge_neutral_pauling_valid",
                "oxidation_states": tuple((int(value) for value in ox_states)),
            }
    if not charge_neutral_seen:
        return {"valid": False, "reason": "charge_neutrality_fail"}
    return {
        "valid": False,
        "reason": "pauling_fail_or_ratio_rejected",
        "pauling_type_error_seen": pauling_error_seen,
    }
