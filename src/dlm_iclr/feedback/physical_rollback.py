"""Select physically screened reconstructions or their paired references."""

from copy import deepcopy

from ..evaluation.physics import record_key

RULE = "restore_confirmed_meta_sun_if_reconstruction_has_no_confirmed_sun_or_meta_sun"


def _check(records, scores, labels):
    identities = []
    for record, score, label in zip(records, scores, labels, strict=True):
        identity = (record["source_id"], record["ordinal"])
        if any((item.get("source_id"), item.get("ordinal")) != identity for item in (score, label)):
            raise ValueError("Physical rollback requires matching source IDs and saved order")
        key = record_key(record)
        if any(item.get("record_key") != key for item in (score, label)):
            raise ValueError("Physical rollback score/label does not belong to the saved geometry")
        if score.get("terminal_status") != label.get("status") or (
            (score.get("terminal_verified") is True) != (label.get("verified") is True)
        ):
            raise ValueError("Physical rollback requires scores from the corresponding physical labels")
        for field in ("strict_sun", "meta_sun"):
            if field not in score or (score[field] is not None and type(score[field]) is not bool):
                raise ValueError("Physical outcome flags must be boolean or unknown")
            if score[field] is True and (
                score.get("terminal_verified") is not True or label.get("status") != "verified"
            ):
                raise ValueError("Physical rollback requires qualified terminal scores")
        if score["strict_sun"] is True and score["meta_sun"] is not True:
            raise ValueError("Confirmed SUN must also be confirmed MSUN")
        identities.append(identity)
    if any(len(identities) != len({identity[i] for identity in identities}) for i in (0, 1)):
        raise ValueError("Physical rollback source/order identities must be unique")
    return identities


def select(
    references,
    reconstructions,
    reference_scores,
    reconstruction_scores,
    reference_labels,
    reconstruction_labels,
):
    """Retain the historical M.S.U.N. rollback rule and every evaluation outcome."""
    sizes = {
        len(x)
        for x in (
            references,
            reconstructions,
            reference_scores,
            reconstruction_scores,
            reference_labels,
            reconstruction_labels,
        )
    }
    if len(sizes) != 1 or not references:
        raise ValueError("Physical rollback requires complete, nonempty paired collections")
    if _check(references, reference_scores, reference_labels) != _check(
        reconstructions, reconstruction_scores, reconstruction_labels
    ):
        raise ValueError("Physical rollback reference and reconstruction order differ")
    records, labels, decisions = [], [], []
    for old, new, before, after, old_label, new_label in zip(
        references,
        reconstructions,
        reference_scores,
        reconstruction_scores,
        reference_labels,
        reconstruction_labels,
        strict=True,
    ):
        restore = before["meta_sun"] is True and not (
            after["strict_sun"] is True or after["meta_sun"] is True
        )
        record, label = (old, old_label) if restore else (new, new_label)
        unresolved = after["strict_sun"] is None or after["meta_sun"] is None
        records.append(deepcopy(record))
        labels.append(deepcopy(label))
        decisions.append(
            {
                "source_id": old["source_id"],
                "ordinal": old["ordinal"],
                "restored_reference": restore,
                "reason": ("reconstruction_unresolved" if unresolved else "discovery_qualification_lost")
                if restore
                else "reconstruction_retained",
                "reference_record_key": record_key(old),
                "reconstruction_record_key": record_key(new),
                "selected_record_key": record_key(record),
                "reference_meta_sun": before["meta_sun"],
                "reconstruction_sun": after["strict_sun"],
                "reconstruction_meta_sun": after["meta_sun"],
                "reference_terminal_status": old_label["status"],
                "reconstruction_terminal_status": new_label["status"],
            }
        )
    return records, labels, decisions
