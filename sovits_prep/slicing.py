import logging
from pathlib import Path
from typing import Dict, List

import ffmpeg
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .utils import compute_hash, ensure_dir, write_csv

logger = logging.getLogger(__name__)


def _chunk_segment(start: float, end: float, min_dur: float, max_dur: float) -> List[tuple[float, float]]:
    chunks: List[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + max_dur, end)
        duration = chunk_end - cursor
        if duration < min_dur:
            break
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _slice_audio(source_wav: Path, out_wav: Path, start: float, end: float) -> None:
    ensure_dir(out_wav.parent)
    duration = end - start
    stream = ffmpeg.input(str(source_wav), ss=start, t=duration)
    stream = ffmpeg.output(stream, str(out_wav), ac=1, ar=16000, format="wav")
    ffmpeg.run(stream, overwrite_output=True, quiet=True)


def _merge_segments(diar_df: pd.DataFrame, gap: float) -> pd.DataFrame:
    merged_records: list[dict] = []
    for (source_id, speaker_id), group in diar_df.groupby(["source_id", "speaker_id"]):
        entries = group.sort_values("start_sec").to_dict("records")
        current = None
        for seg in entries:
            if current is None:
                current = seg
                continue
            if seg["start_sec"] - current["end_sec"] <= gap:
                current["end_sec"] = max(current["end_sec"], seg["end_sec"])
                current["duration_sec"] = current["end_sec"] - current["start_sec"]
                current["overlap"] = bool(current.get("overlap", False) or seg.get("overlap", False))
                current["overlap_ratio"] = max(
                    float(current.get("overlap_ratio", 0.0)), float(seg.get("overlap_ratio", 0.0))
                )
            else:
                merged_records.append(current)
                current = seg
        if current is not None:
            merged_records.append(current)
    return pd.DataFrame(merged_records)


def run_slicing(sources_df: pd.DataFrame, diar_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    meta_dir = config.out_root / "metadata"
    slices_path = meta_dir / "slices.csv"
    stage_hash = compute_hash(
        {
            "min_duration_sec": config.min_duration_sec,
            "max_duration_sec": config.max_duration_sec,
            "merge_gap_sec": config.merge_gap_sec,
        }
    )

    ensure_dir(meta_dir)
    segments_dir = config.out_root / "segments"
    ensure_dir(segments_dir)

    source_map: Dict[int, Path] = {
        int(row["source_id"]): config.out_root / row["audio_path"] for _, row in sources_df.iterrows()
    }

    diar_df = diar_df.copy()
    diar_df["duration_sec"] = pd.to_numeric(diar_df.get("duration_sec", 0), errors="coerce")
    diar_df["start_sec"] = pd.to_numeric(diar_df.get("start_sec", 0), errors="coerce")
    diar_df["end_sec"] = pd.to_numeric(diar_df.get("end_sec", 0), errors="coerce")
    diar_df["speaker_id"] = diar_df.get("speaker_id", "").astype(str).str.strip()

    logger.info("Diarization rows before slicing: %d", len(diar_df))
    diar_valid = diar_df[(diar_df["duration_sec"] > 0) & (diar_df["speaker_id"].str.len() > 0)]
    diar_sorted = diar_valid.sort_values(["source_id", "start_sec"])
    logger.info("Diarization rows after basic filter: %d", len(diar_sorted))
    diar_merged = _merge_segments(diar_sorted, config.merge_gap_sec)
    if diar_merged.empty:
        logger.warning(
            "Merge produced zero segments (input rows: %d). Falling back to unmerged diarization.",
            len(diar_sorted),
        )
        diar_merged = diar_sorted.copy()
    for col in ("start_sec", "end_sec", "duration_sec"):
        if col not in diar_merged.columns:
            diar_merged[col] = 0.0
    diar_merged["start_sec"] = pd.to_numeric(diar_merged["start_sec"], errors="coerce")
    diar_merged["end_sec"] = pd.to_numeric(diar_merged["end_sec"], errors="coerce")
    diar_merged["duration_sec"] = pd.to_numeric(diar_merged["duration_sec"], errors="coerce")
    diar_merged["duration_sec"] = diar_merged["duration_sec"].fillna(diar_merged["end_sec"] - diar_merged["start_sec"])
    diar_merged = diar_merged.dropna(subset=["start_sec", "end_sec", "duration_sec"])
    diar_merged = diar_merged[diar_merged["duration_sec"] > 0]
    merged_total = len(diar_merged)
    ge_min_mask = diar_merged["duration_sec"] >= config.min_duration_sec
    merged_ge_min = int(ge_min_mask.sum())
    diar_merged = diar_merged[ge_min_mask]
    logger.info("Merged diarization segments: %d (>=min_duration_sec: %d)", merged_total, merged_ge_min)

    records: list[dict] = []
    slice_counter = 0

    for _, row in tqdm(diar_merged.iterrows(), total=len(diar_merged), desc="Slicing", unit="seg"):
        source_audio = source_map.get(int(row["source_id"]))
        if source_audio is None or not source_audio.exists():
            tqdm.write(f"[slicing] skip source_id={row['source_id']} missing audio")
            continue

        diar_hash = row.get("diarization_config_hash", "")

        chunks = _chunk_segment(
            float(row["start_sec"]),
            float(row["end_sec"]),
            config.min_duration_sec,
            config.max_duration_sec,
        )
        if not chunks:
            tqdm.write(
                f"[slicing] skip source_id={row['source_id']} speaker={row['speaker_id']} "
                f"start={row['start_sec']:.2f} end={row['end_sec']:.2f} (too short)"
            )
        for start_sec, end_sec in chunks:
            wav_name = f"{row['speaker_id']}_{slice_counter:06d}.wav"
            wav_path = segments_dir / str(row["speaker_id"]) / wav_name
            try:
                _slice_audio(source_audio, wav_path, start_sec, end_sec)
            except Exception as exc:  # noqa: BLE001
                tqdm.write(
                    f"[slicing] ffmpeg failed for slice_id={slice_counter} src={source_audio.name}: {exc}"
                )
                continue

            records.append(
                {
                    "slice_id": slice_counter,
                    "source_id": int(row["source_id"]),
                    "speaker_id": str(row["speaker_id"]),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                    "wav_path": wav_path.relative_to(config.out_root).as_posix(),
                    "is_multi_speaker": bool(row.get("overlap", False)),
                    "diarization_config_hash": diar_hash,
                    "slice_config_hash": stage_hash,
                }
            )
            slice_counter += 1

    df = pd.DataFrame(records)
    write_csv(df, slices_path)
    return df
