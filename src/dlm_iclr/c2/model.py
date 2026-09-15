"""A single C2 entry point composing the already validated editing and repair steps."""

from copy import deepcopy


class C2:
    def __init__(self, editor, onepass):
        self.editor = editor
        self.onepass = onepass

    def edit_many(self, plans, refined, *, bundles=None):
        edits, features = self.editor.edit(plans, refined, retain_features=True)
        prepared = (
            [{"plan": plan, "F": state} for plan, state in zip(plans, refined)]
            if bundles is None
            else deepcopy(bundles)
        )
        for bundle, edit in zip(prepared, edits):
            bundle["E"] = edit
        return self.onepass.repair(prepared, keep_features=features["keep"])

    def edit(self, plan, refined):
        """Return the full trace for one Plan and its continuous F800 wrapper."""
        return self.edit_many([plan], [refined])[0]
