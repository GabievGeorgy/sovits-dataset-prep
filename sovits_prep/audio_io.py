from pathlib import Path
import shutil
from typing import List

import ffmpeg
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .utils import ensure_dir, read_csv, relative_to_root, wav_duration_seconds, write_csv


INPUT_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wav", ".flac", ".mp3", ".m4a"}


def find_input_files(input_dir: Path, exclude_root: Path | None = None) -> List[Path]:
    if input_dir.is_file():
        return [input_dir] if input_dir.suffix.lower() in INPUT_EXTS else []
    paths: list[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in INPUT_EXTS:
            if exclude_root and path.is_relative_to(exclude_root):
                continue
            paths.append(path)
    return sorted(paths)


def _is_mono_16k_wav(path: Path) -> bool:
    try:
        info = ffmpeg.probe(str(path))
        streams = info.get("streams", [])
        if not streams:
            return False
        audio_stream = streams[0]
        return (
            path.suffix.lower() == ".wav"
            and int(audio_stream.get("channels", 0)) == 1
            and int(audio_stream.get("sample_rate", 0)) == 16000
        )
    except ffmpeg.Error:
        return False


def prepare_audio(input_path: Path, out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    if _is_mono_16k_wav(input_path):
        shutil.copyfile(input_path, out_path)
        return out_path

    stream = ffmpeg.input(str(input_path))
    stream = ffmpeg.output(stream, str(out_path), ac=1, ar=16000, format="wav")
    ffmpeg.run(stream, overwrite_output=True, quiet=True)
    return out_path


def prepare_sources(config: PipelineConfig) -> pd.DataFrame:
    meta_dir = config.out_root / "metadata"
    sources_path = meta_dir / "sources.csv"

    input_files = find_input_files(config.input_dir, exclude_root=config.out_root)
    records = []
    raw_dir = config.out_root / "raw_audio"
    ensure_dir(raw_dir)

    for idx, src_path in enumerate(tqdm(input_files, desc="Preparing audio", unit="file")):
        out_path = raw_dir / f"source_{idx:04d}.wav"
        try:
            processed = prepare_audio(src_path, out_path)
            duration = wav_duration_seconds(processed)
            records.append(
                {
                    "source_id": idx,
                    "input_path": src_path.as_posix(),
                    "audio_path": relative_to_root(processed, config.out_root),
                    "duration_sec": duration,
                    "status": "ok",
                    "error_message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "source_id": idx,
                    "input_path": src_path.as_posix(),
                    "audio_path": "",
                    "duration_sec": 0.0,
                    "status": "error",
                    "error_message": str(exc),
                }
            )

    df = pd.DataFrame(records)
    write_csv(df, sources_path)
    return df
