import logging
from pathlib import Path
from typing import Optional

import click
from click.core import ParameterSource

from sovits_prep.config import PipelineConfig
from sovits_prep.pipeline import run_pipeline


def _split_csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    items = [v for v in items if v]
    return items or None


def _log_defaulted_params(ctx: click.Context) -> None:
    defaulted: dict[str, object] = {}
    for key in (
        "hf_token",
        "min_duration_sec",
        "max_duration_sec",
        "min_words",
        "min_quality_mos",
        "drop_multi_speaker",
        "merge_gap_sec",
        "diar_min_speech_sec",
        "diar_min_pause_sec",
        "allowed_speakers",
        "min_speakers",
        "max_speakers",
        "device",
        "view_name",
    ):
        if ctx.get_parameter_source(key) == ParameterSource.DEFAULT:
            defaulted[key] = ctx.params.get(key)
    if defaulted:
        logging.info(
            "Using default parameters: %s",
            ", ".join(f"{k}={v}" for k, v in defaulted.items()),
        )


@click.command()
@click.option("--input", "input_dir", type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path), required=True)
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--view-name", type=str, required=True)
@click.option("--hf-token", type=str, default=None, help="HuggingFace token")
@click.option("--min-duration-sec", type=float, default=3.0, show_default=True)
@click.option("--max-duration-sec", type=float, default=15.0, show_default=True)
@click.option("--min-words", type=int, default=2, show_default=True)
@click.option("--min-quality-mos", type=float, default=3.5, show_default=True)
@click.option("--drop-multi-speaker", is_flag=True, default=False, show_default=True)
@click.option("--merge-gap-sec", type=float, default=2.0, show_default=True)
@click.option(
    "--diar-min-speech-sec",
    type=float,
    default=1.5,
    show_default=True,
    help="Diarization: minimum speech span the model keeps",
)
@click.option(
    "--diar-min-pause-sec",
    type=float,
    default=1.0,
    show_default=True,
    help="Diarization: minimum pause between speech spans",
)
@click.option("--allowed-speakers", type=str, default=None, help="Comma-separated list")
@click.option("--min-speakers", type=int, default=None, show_default=True, help="Minimum expected speakers")
@click.option("--max-speakers", type=int, default=None, show_default=True, help="Maximum expected speakers")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    default="auto",
    show_default=True,
)
@click.pass_context
def main(
    ctx: click.Context,
    input_dir: Path,
    out_root: Path,
    view_name: str,
    hf_token: Optional[str],
    min_duration_sec: float,
    max_duration_sec: float,
    min_words: int,
    min_quality_mos: float,
    drop_multi_speaker: bool,
    merge_gap_sec: float,
    diar_min_speech_sec: float,
    diar_min_pause_sec: float,
    allowed_speakers: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    device: str,
) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
    _log_defaulted_params(ctx)
    config = PipelineConfig(
        input_dir=input_dir,
        out_root=out_root,
        view_name=view_name,
        hf_token=hf_token,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        min_words=min_words,
        min_quality_mos=min_quality_mos,
        drop_multi_speaker=drop_multi_speaker,
        merge_gap_sec=merge_gap_sec,
        diar_min_speech_sec=diar_min_speech_sec,
        diar_min_pause_sec=diar_min_pause_sec,
        allowed_speakers=_split_csv(allowed_speakers),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device.lower(),
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
