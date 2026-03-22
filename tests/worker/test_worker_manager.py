from pathlib import Path

from gpustack.config.config import Config
from gpustack.worker.worker_manager import (
    DIRECT_PROCESS_MODE_LABEL,
    WorkerManager,
)


def make_config(tmp_path: Path, direct_process_mode: bool) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path),
        server_url="http://127.0.0.1:30080",
        direct_process_mode=direct_process_mode,
        worker_name="worker-1",
    )


def test_builtin_labels_include_direct_process_signal_when_enabled(tmp_path: Path):
    manager = object.__new__(WorkerManager)
    manager._cfg = make_config(tmp_path, direct_process_mode=True)

    labels = manager._ensure_builtin_labels()

    assert labels[DIRECT_PROCESS_MODE_LABEL] == "true"


def test_builtin_labels_omit_direct_process_signal_when_disabled(tmp_path: Path):
    manager = object.__new__(WorkerManager)
    manager._cfg = make_config(tmp_path, direct_process_mode=False)

    labels = manager._ensure_builtin_labels()

    assert DIRECT_PROCESS_MODE_LABEL not in labels
