import numpy as np
import soundfile as sf
from pathlib import Path

from sovits_prep.audio_io import prepare_sources
from sovits_prep.config import PipelineConfig


def _make_wav(path: Path, sr: int = 16000, duration_sec: float = 0.1) -> Path:
    samples = int(sr * duration_sec)
    data = np.zeros(samples, dtype=np.float32)
    sf.write(path, data, sr)
    return path


def _base_config(input_path: Path, out_root: Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=input_path,
        out_root=out_root,
        view_name="test",
        hf_token=None,
    )


def test_prepare_sources_accepts_file(tmp_path: Path) -> None:
    input_file = _make_wav(tmp_path / "one.wav")
    out_root = tmp_path / "out"
    cfg = _base_config(input_file, out_root)

    df = prepare_sources(cfg)

    assert len(df) == 1
    assert df.loc[0, "status"] == "ok"
    raw_path = out_root / df.loc[0, "audio_path"]
    assert raw_path.exists()
    assert raw_path.is_file()


def test_prepare_sources_accepts_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _make_wav(input_dir / "a.wav")
    _make_wav(input_dir / "b.wav")

    out_root = tmp_path / "out"
    cfg = _base_config(input_dir, out_root)

    df = prepare_sources(cfg)

    assert len(df) == 2
    assert set(df["status"]) == {"ok"}
    for rel_path in df["audio_path"]:
        full_path = out_root / rel_path
        assert full_path.exists()
        assert full_path.is_file()
