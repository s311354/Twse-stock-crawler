"""Shared failure contract for a job whose pinned execution version changed."""
import json


class WorkVersionChanged(ValueError):
    """Keep the legacy error code, with bounded, non-secret drift diagnostics."""

    def __init__(self, changes):
        self.diagnostics = {"schema": "work-version-drift/v1", "changes": changes,
                            "action": "RESTART_WITH_NEW_JOB"}
        super().__init__("REPLAY_VERSION_CHANGED_DURING_JOB: " + json.dumps(
            self.diagnostics, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def __reduce__(self):
        # ProcessPoolExecutor must reconstruct from changes, not the already
        # formatted exception message. Preserve checkpoint/notes as state too.
        return type(self), (self.diagnostics["changes"],), self.__dict__
