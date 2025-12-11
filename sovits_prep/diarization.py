import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from tqdm import tqdm

from .config import PipelineConfig
from .utils import compute_hash, ensure_dir, write_csv

logger = logging.getLogger(__name__)
DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-community-1"


def _compute_overlap_ratio(segments: List[Dict]) -> List[Dict]:
    for idx, seg in enumerate(segments):
        overlap_duration = 0.0
        for jdx, other in enumerate(segments):
            if idx == jdx:
                continue
            if other["speaker_id"] == seg["speaker_id"]:
                continue
            start = max(seg["start_sec"], other["start_sec"])
            end = min(seg["end_sec"], other["end_sec"])
            overlap = max(0.0, end - start)
            overlap_duration += overlap
        ratio = overlap_duration / seg["duration_sec"] if seg["duration_sec"] > 0 else 0.0
        seg["overlap_ratio"] = min(1.0, ratio)
        seg["overlap"] = seg["overlap_ratio"] > 0
    return segments


def _result_to_annotation(result) -> Annotation:
    if isinstance(result, Annotation):
        return result

    speaker_diarization = getattr(result, "speaker_diarization", None)
    if isinstance(speaker_diarization, Annotation):
        return speaker_diarization

    annotation = getattr(result, "annotation", None)
    if isinstance(annotation, Annotation):
        return annotation

    diarization = getattr(result, "diarization", None)
    if isinstance(diarization, Annotation):
        return diarization

    raise TypeError(f"Unsupported annotation type: {type(result)}")


def _iter_segments(annotation: Annotation):
    if hasattr(annotation, "itertracks"):
        return annotation.itertracks(yield_label=True)
    if hasattr(annotation, "iter_segments"):  # legacy
        return annotation.iter_segments(yield_label=True)
    raise AttributeError("Annotation object has no itertracks/iter_segments")


def run_diarization(
    sources_df: pd.DataFrame, config: PipelineConfig, device: "torch.device", device_tag: str
) -> pd.DataFrame:
    meta_dir = config.out_root / "metadata"
    diar_path = meta_dir / "diarization.csv"
    stage_hash = compute_hash(
        {
            "diarization_model_name": DIARIZATION_MODEL_NAME,
            "device": device_tag,
            "iter_adapter": "v2",
            "min_speakers": config.min_speakers,
            "max_speakers": config.max_speakers,
            "skip_overlap": True,
            "min_duration_on": config.diar_min_speech_sec,
            "min_duration_off": config.diar_min_pause_sec,
        }
    )

    ensure_dir(meta_dir)
    hf_token = config.hf_token or os.getenv("HF_TOKEN")
    diar_model = Pipeline.from_pretrained(DIARIZATION_MODEL_NAME, token=hf_token)
    diar_model.to(device)

    records: list[dict] = []
    ok_sources = sources_df[sources_df["status"] == "ok"]

    for _, row in tqdm(ok_sources.iterrows(), total=len(ok_sources), desc="Diarization", unit="file"):
        audio_path = config.out_root / row["audio_path"]
        try:
            tqdm.write(
                f"[diar] start source_id={int(row['source_id'])} "
                f"path={audio_path.name} dur={float(row['duration_sec']):.1f}s"
            )
            t0 = time.time()
            diar_result = diar_model(
                str(audio_path),
                skip_overlap=True,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
                min_duration_on=config.diar_min_speech_sec,
                min_duration_off=config.diar_min_pause_sec,
            )
            annotation = _result_to_annotation(diar_result)
            iterator = _iter_segments(annotation)
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "source_id": int(row["source_id"]),
                    "speaker_id": "",
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": 0.0,
                    "overlap": True,
                    "overlap_ratio": 1.0,
                    "diarization_config_hash": stage_hash,
                    "error_message": f"{exc} | type={type(exc).__name__} | result_type={type(diar_result)}",
                }
            )
            tqdm.write(
                f"[diar] failed to diarize source_id={int(row['source_id'])}: {exc} "
                f"(type={type(exc).__name__}, result={type(diar_result)})"
            )
            continue

        segments: List[Dict] = []
        try:
            for item in iterator:
                if len(item) == 3:
                    segment, track, label = item
                elif len(item) == 2:
                    segment, label = item
                    track = None
                else:
                    raise ValueError("Unexpected iterator output")
                start = float(segment.start)
                end = float(segment.end)
                segments.append(
                    {
                        "source_id": int(row["source_id"]),
                        "speaker_id": str(label),
                        "start_sec": start,
                        "end_sec": end,
                        "duration_sec": end - start,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "source_id": int(row["source_id"]),
                    "speaker_id": "",
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": 0.0,
                    "overlap": True,
                    "overlap_ratio": 1.0,
                    "diarization_config_hash": stage_hash,
                    "error_message": str(exc),
                }
            )
            tqdm.write(
                f"[diar] failed to read segments for source_id={int(row['source_id'])}: {exc} "
                f"(type={type(annotation)})"
            )
            continue

        if segments:
            tqdm.write(
                f"[diar] done source_id={int(row['source_id'])} "
                f"segments={len(segments)} elapsed={time.time()-t0:.1f}s"
            )
        else:
            tqdm.write(
                f"[diar] done source_id={int(row['source_id'])} no segments produced "
                f"elapsed={time.time()-t0:.1f}s"
            )

        segments = _compute_overlap_ratio(segments)
        for seg in segments:
            seg["diarization_config_hash"] = stage_hash
            records.append(seg)

    df = pd.DataFrame(records)
    write_csv(df, diar_path)
    return df
