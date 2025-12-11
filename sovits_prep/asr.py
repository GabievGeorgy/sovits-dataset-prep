from pathlib import Path
from typing import Tuple

import gigaam
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .utils import compute_hash, ensure_dir, write_csv

ASR_MODEL_NAME = "v3_e2e_rnnt"


def _predict_text(model, wav_path: Path) -> Tuple[str, str]:
    if hasattr(model, "transcribe"):
        result = model.transcribe(str(wav_path))
    elif hasattr(model, "predict"):
        result = model.predict(str(wav_path))
    elif callable(model):
        result = model(str(wav_path))
    else:
        raise RuntimeError("ASR model does not expose a known interface")

    if isinstance(result, tuple) and len(result) >= 1:
        result = result[0]

    if isinstance(result, dict):
        text = result.get("text") or result.get("transcription") or result.get("transcript") or ""
        text_raw = result.get("text_raw") or text
    else:
        text = str(result) if result is not None else ""
        text_raw = text

    return text.strip(), text_raw.strip()


def run_asr_on_slices(slices_df: pd.DataFrame, config: PipelineConfig, device: str) -> pd.DataFrame:
    meta_dir = config.out_root / "metadata"
    asr_path = meta_dir / "asr.csv"
    stage_hash = compute_hash(
        {
            "asr_model_name": ASR_MODEL_NAME,
            "device": device,
        }
    )

    ensure_dir(meta_dir)
    model = gigaam.load_model(ASR_MODEL_NAME, device=device)

    records: list[dict] = []
    for _, row in tqdm(slices_df.iterrows(), total=len(slices_df), desc="ASR", unit="slice"):
        wav_path = config.out_root / row["wav_path"]
        try:
            text, text_raw = _predict_text(model, wav_path)
        except Exception:  # noqa: BLE001
            text, text_raw = "", ""

        num_words = len(text.split()) if text else 0
        num_chars = len(text) if text else 0

        records.append(
            {
                "slice_id": int(row["slice_id"]),
                "text": text,
                "text_raw": text_raw,
                "num_chars": num_chars,
                "num_words": num_words,
                "asr_model_name": ASR_MODEL_NAME,
                "asr_config_hash": stage_hash,
            }
        )

    df = pd.DataFrame(records)
    write_csv(df, asr_path)
    return df
