from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
try:
    import torchaudio
except Exception:  # noqa: BLE001
    torchaudio = None
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
from tqdm import tqdm

from .config import PipelineConfig
from .utils import compute_hash, ensure_dir, write_csv


def _load_audio(wav_path: Path) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(wav_path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tensor = torch.tensor(audio, dtype=torch.float32)
    return tensor, sr


def _maybe_resample(audio_t: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if not isinstance(src_sr, int) or not isinstance(dst_sr, int):
        return audio_t
    if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
        return audio_t
    if torchaudio is None:
        return audio_t
    try:
        audio_2d = audio_t.unsqueeze(0)
        resampled = torchaudio.functional.resample(audio_2d, src_sr, dst_sr)
        return resampled.squeeze(0)
    except Exception:  # noqa: BLE001
        return audio_t


def run_quality_on_slices(
    slices_df: pd.DataFrame, config: PipelineConfig, device: torch.device, device_tag: str
) -> pd.DataFrame:
    meta_dir = config.out_root / "metadata"
    quality_path = meta_dir / "quality.csv"

    ensure_dir(meta_dir)
    metric_fs = int(config.sample_rate)
    try:
        metric = NonIntrusiveSpeechQualityAssessment(fs=metric_fs).to(device)
    except Exception:  # noqa: BLE001
        metric_fs = 16000
        metric = NonIntrusiveSpeechQualityAssessment(fs=metric_fs).to(device)
    stage_hash = compute_hash(
        {
            "metric": "torchmetrics_nonintrusive_speech_quality",
            "device": device_tag,
            "output_sr": int(config.sample_rate),
            "metric_fs": int(metric_fs),
        }
    )

    records: list[dict] = []
    for _, row in tqdm(slices_df.iterrows(), total=len(slices_df), desc="Quality", unit="slice"):
        wav_path = config.out_root / row["wav_path"]
        try:
            audio_t, sr = _load_audio(wav_path)
            if audio_t.numel() == 0:
                raise ValueError("Empty audio tensor")
            audio_t = _maybe_resample(audio_t, int(sr), metric_fs)

            audio_t = audio_t.to(device)
            scores = metric(audio_t)
            mos, noisiness, discontinuity, coloration, loudness = scores.tolist()
        except Exception:  # noqa: BLE001
            mos = noisiness = discontinuity = coloration = loudness = 0.0

        records.append(
            {
                "slice_id": int(row["slice_id"]),
                "quality_mos": float(mos),
                "quality_noisiness": float(noisiness),
                "quality_discontinuity": float(discontinuity),
                "quality_coloration": float(coloration),
                "quality_loudness": float(loudness),
                "quality_config_hash": stage_hash,
            }
        )

    df = pd.DataFrame(records)
    write_csv(df, quality_path)
    return df
