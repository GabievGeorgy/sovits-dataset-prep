from pathlib import Path
import pandas as pd

from .config import PipelineConfig
from .utils import ensure_dir, read_csv, write_csv


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.reindex(columns=columns)


def _empty_segments_df() -> pd.DataFrame:
    columns = [
        "segment_id",
        "slice_id",
        "source_id",
        "source_audio_path",
        "speaker_id",
        "wav_path",
        "start_sec",
        "end_sec",
        "duration_sec",
        "is_multi_speaker",
        "text",
        "text_raw",
        "num_chars",
        "num_words",
        "emotion_label",
        "emotion_positive",
        "emotion_angry",
        "emotion_sad",
        "emotion_neutral",
        "quality_mos",
        "quality_noisiness",
        "quality_discontinuity",
        "quality_coloration",
        "quality_loudness",
        "language",
        "diarization_config_hash",
        "slice_config_hash",
        "asr_config_hash",
        "emotion_config_hash",
        "quality_config_hash",
    ]
    return pd.DataFrame(columns=columns)


def build_segments_table(meta_dir: Path) -> pd.DataFrame:
    sources = _ensure_columns(
        read_csv(meta_dir / "sources.csv"),
        ["source_id", "audio_path", "duration_sec", "status", "error_message"],
    )
    slices = _ensure_columns(
        read_csv(meta_dir / "slices.csv"),
        [
            "slice_id",
            "source_id",
            "speaker_id",
            "start_sec",
            "end_sec",
            "duration_sec",
            "wav_path",
            "is_multi_speaker",
            "slice_config_hash",
            "diarization_config_hash",
        ],
    )
    asr_df = _ensure_columns(
        read_csv(meta_dir / "asr.csv"),
        ["slice_id", "text", "text_raw", "num_chars", "num_words", "asr_model_name", "asr_config_hash"],
    )
    emo_df = _ensure_columns(
        read_csv(meta_dir / "emotions.csv"),
        [
            "slice_id",
            "emotion_label",
            "emotion_positive",
            "emotion_angry",
            "emotion_sad",
            "emotion_neutral",
            "emotion_config_hash",
        ],
    )
    quality_df = _ensure_columns(
        read_csv(meta_dir / "quality.csv"),
        [
            "slice_id",
            "quality_mos",
            "quality_noisiness",
            "quality_discontinuity",
            "quality_coloration",
            "quality_loudness",
            "quality_config_hash",
        ],
    )

    if slices.empty:
        segments_path = meta_dir / "segments_full.csv"
        segments_df = _empty_segments_df()
        write_csv(segments_df, segments_path)
        return segments_df

    df = slices.merge(asr_df, on="slice_id", how="left")
    df = df.merge(emo_df, on="slice_id", how="left")
    df = df.merge(quality_df, on="slice_id", how="left")
    df = df.merge(
        sources[["source_id", "audio_path"]].rename(columns={"audio_path": "source_audio_path"}),
        on="source_id",
        how="left",
    )

    df = df.copy()
    df.insert(0, "segment_id", df["slice_id"])
    df["language"] = "ru"

    segments_path = meta_dir / "segments_full.csv"
    write_csv(df, segments_path)
    return df


def _apply_filters(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    filtered = df.copy()
    total = len(filtered)
    stats: list[str] = []

    duration_mask = (filtered["duration_sec"] >= config.min_duration_sec) & (
        filtered["duration_sec"] <= config.max_duration_sec
    )
    filtered = filtered[duration_mask]
    if total:
        stats.append(f"duration: {len(filtered)}/{total} kept ({len(filtered)/total:.0%})")
    total = len(filtered)

    words_mask = filtered["num_words"].fillna(0) >= config.min_words
    filtered = filtered[words_mask]
    if total:
        stats.append(f"words: {len(filtered)}/{total} kept ({len(filtered)/total:.0%})")
    total = len(filtered)

    quality_mask = filtered["quality_mos"].fillna(0) >= config.min_quality_mos
    filtered = filtered[quality_mask]
    if total:
        stats.append(f"quality: {len(filtered)}/{total} kept ({len(filtered)/total:.0%})")
    total = len(filtered)

    if config.drop_multi_speaker and "is_multi_speaker" in filtered.columns:
        ms_mask = ~filtered["is_multi_speaker"].fillna(False)
        filtered = filtered[ms_mask]
        if total:
            stats.append(f"multi_speaker: {len(filtered)}/{total} kept ({len(filtered)/total:.0%})")
        total = len(filtered)

    if config.allowed_speakers:
        spk_mask = filtered["speaker_id"].isin(config.allowed_speakers)
        filtered = filtered[spk_mask]
        if total:
            stats.append(f"allowed_speakers: {len(filtered)}/{total} kept ({len(filtered)/total:.0%})")

    if stats:
        print("[view][filter] " + " | ".join(stats))

    return filtered


def create_view(segments_full: pd.DataFrame, config: PipelineConfig) -> None:
    view_dir = config.out_root / "views" / config.view_name
    ensure_dir(view_dir)

    view_df = _apply_filters(segments_full, config)
    if view_df.empty:
        print("[view] no rows after filtering; segments.csv and train.list will be empty")
    write_csv(view_df, view_dir / "segments.csv")

    list_path = view_dir / "train.list"
    lines = []
    for _, row in view_df.iterrows():
        wav_path = (config.out_root / row["wav_path"]).as_posix()
        speaker_id = row["speaker_id"]
        text = row.get("text", "")
        lines.append(f"{wav_path}|{speaker_id}|ru|{text}")

    list_path.write_text("\n".join(lines), encoding="utf-8")

    for speaker_id, group in view_df.groupby("speaker_id"):
        if pd.isna(speaker_id) or str(speaker_id).strip() == "":
            continue
        safe_speaker = str(speaker_id).replace("/", "_").replace("\\", "_")
        spk_lines = []
        for _, row in group.iterrows():
            wav_path = (config.out_root / row["wav_path"]).as_posix()
            text = row.get("text", "")
            spk_lines.append(f"{wav_path}|{speaker_id}|ru|{text}")
        spk_path = view_dir / f"train_{safe_speaker}.list"
        spk_path.write_text("\n".join(spk_lines), encoding="utf-8")
