import json
import os

import pytest

from infrastructure.run_lock import ProcessRunLock
from infrastructure.run_lock import RunAlreadyActiveError


def test_process_run_lock_rejects_overlap_and_reports_owner(tmp_path) -> None:
    path = tmp_path / "analysis.lock"
    first = ProcessRunLock(path)
    second = ProcessRunLock(path)

    with first:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["pid"] == os.getpid()
        assert metadata["host"]
        assert metadata["started_at"]

        with pytest.raises(RunAlreadyActiveError) as error:
            second.acquire()

        assert "pid={}".format(os.getpid()) in str(error.value)
        assert "started_at=" in str(error.value)

    with second:
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_process_run_lock_releases_after_exception(tmp_path) -> None:
    path = tmp_path / "analysis.lock"

    with pytest.raises(RuntimeError, match="analysis failed"):
        with ProcessRunLock(path):
            raise RuntimeError("analysis failed")

    with ProcessRunLock(path):
        pass
