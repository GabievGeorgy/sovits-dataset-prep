from pathlib import Path
from typing import Dict

import gigaam
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .utils import compute_hash, ensure_dir, write_csv


EMO_KEYS = ["positive", "angry", "sad", "neutral"]
EMOTION_MODEL_NAME = "emo"


def _extract_scores(result: Dict) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if isinstance(result, dict):
        if "scores" in result and isinstance(result["scores"], dict):
            scores.update({k: float(v) for k, v in result["scores"].items() if k in EMO_KEYS})
        for key in EMO_KEYS:
            if key in result:
                scores[key] = float(result[key])
        if "probabilities" in result and isinstance(result["probabilities"], dict):
            scores.update({k: float(v) for k, v in result["probabilities"].items() if k in EMO_KEYS})
    elif isinstance(result, (list, tuple)) and len(result) >= len(EMO_KEYS):
        for key, val in zip(EMO_KEYS, result):
            scores[key] = float(val)
    return scores


def _predict_emotion(model, wav_path: Path) -> Dict[str, float]:
    if hasattr(model, "predict"):
        result = model.predict(str(wav_path))
    elif hasattr(model, "infer"):
        result = model.infer(str(wav_path))
    elif callable(model):
        result = model(str(wav_path))
    else:
        raise RuntimeError("Emotion model does not expose a known interface")
    scores = _extract_scores(result)
    if not scores:
        scores = {k: 0.0 for k in EMO_KEYS}
    return scores


def run_emotion_on_slices(slices_df: pd.DataFrame, config: PipelineConfig, device: str) -> pd.DataFrame:
    meta_dir = config.out_root / "metadata"
    emo_path = meta_dir / "emotions.csv"
    stage_hash = compute_hash(
        {
            "emotion_model_name": EMOTION_MODEL_NAME,
            "device": device,
            "sample_rate": int(config.sample_rate),
        }
    )

    ensure_dir(meta_dir)
    model = gigaam.load_model(EMOTION_MODEL_NAME, device=device)

    records: list[dict] = []
    for _, row in tqdm(slices_df.iterrows(), total=len(slices_df), desc="Emotion", unit="slice"):
        wav_path = config.out_root / row["wav_path"]
        try:
            scores = _predict_emotion(model, wav_path)
        except Exception:  # noqa: BLE001
            scores = {k: 0.0 for k in EMO_KEYS}

        label = max(scores, key=scores.get) if scores else "neutral"

        records.append(
            {
                "slice_id": int(row["slice_id"]),
                "emotion_label": label,
                "emotion_positive": scores.get("positive", 0.0),
                "emotion_angry": scores.get("angry", 0.0),
                "emotion_sad": scores.get("sad", 0.0),
                "emotion_neutral": scores.get("neutral", 0.0),
                "emotion_config_hash": stage_hash,
            }
        )

    df = pd.DataFrame(records)
    write_csv(df, emo_path)
    return df
