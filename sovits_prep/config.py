from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    input_dir: Path
    out_root: Path
    view_name: str
    hf_token: Optional[str]

    device: str = "auto"

    # Filtering
    min_duration_sec: float = 3.0
    max_duration_sec: float = 15.0
    min_words: int = 2
    min_quality_mos: float = 3.5
    drop_multi_speaker: bool = False
    allowed_speakers: Optional[List[str]] = None
    merge_gap_sec: float = 1.5

    # Models
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    diar_min_speech_sec: float = 3.0
    diar_min_pause_sec: float = 1.5
