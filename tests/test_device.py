import torch
import pytest

from sovits_prep.utils import resolve_device


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device_str, device_obj, note = resolve_device("auto")
    assert device_str == "cpu"
    assert str(device_obj) == "cpu"
    assert note and "not available" in note.lower()


def test_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
