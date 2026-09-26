"""Paper-facing names with stable checkpoint and configuration identifiers."""

MODULE_NAMES = {"constructor": "b0", "periodic": "c1", "feedback": "c2"}
FEEDBACK_STAGES = {"reconstruction": "editor", "refit": "light", "verifier": "value"}


def module_key(name):
    return MODULE_NAMES.get(name, name)


def module_name(key):
    return next((name for name, value in MODULE_NAMES.items() if value == key), key)


def feedback_stage_key(name):
    return FEEDBACK_STAGES.get(name, name)
