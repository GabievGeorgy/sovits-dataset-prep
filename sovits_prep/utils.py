import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from pandas.errors import EmptyDataError
import soundfile as sf
import torch
import shutil


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_hash(payload: Any) -> str:
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple, set)):
            return [_serialize(v) for v in obj]
        return obj

    normalized = _serialize(payload)
    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def wav_duration_seconds(path: Path) -> float:
    info = sf.info(path)
    if info.frames == 0 or info.samplerate == 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def reset_dir(path: Path) -> None:
    """Remove a directory if it exists and recreate it."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def filter_allowed(values: Iterable[str], allowed: Optional[Iterable[str]]) -> set[str]:
    if allowed is None:
        return set(values)
    allowed_set = {v.strip() for v in allowed if v.strip()}
    return {v for v in values if v in allowed_set}


def resolve_device(device_pref: str) -> tuple[str, torch.device, Optional[str]]:
    pref = device_pref.lower().strip()
    if pref == "cpu":
        return "cpu", torch.device("cpu"), None
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda"), None
        raise RuntimeError("CUDA requested but not available")
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda"), None
    return "cpu", torch.device("cpu"), "CUDA not available; using CPU."
