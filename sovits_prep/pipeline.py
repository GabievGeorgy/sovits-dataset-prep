import logging

from . import asr, audio_io, diarization, emotion, metadata_io, quality, slicing
from .config import PipelineConfig
from .utils import ensure_dir, resolve_device, reset_dir

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig) -> None:
    device_str, device_obj, device_note = resolve_device(config.device)
    logger.info("Using device: %s", device_str)
    if device_note:
        logger.warning(device_note)

    if config.input_dir.resolve() == config.out_root.resolve():
        raise ValueError("input_dir and out_root must differ; refusing to clear outputs")
    logger.info("Clearing output directory: %s", config.out_root)
    reset_dir(config.out_root)
    ensure_dir(config.out_root)
    ensure_dir(config.out_root / "metadata")
    ensure_dir(config.out_root / "views")

    sources_df = audio_io.prepare_sources(config)
    logger.info("Sources prepared: %d ok / %d total", (sources_df["status"] == "ok").sum(), len(sources_df))
    diar_df = diarization.run_diarization(sources_df, config, device=device_obj, device_tag=device_str)
    logger.info("Diarization segments: %d", len(diar_df))
    slices_df = slicing.run_slicing(sources_df, diar_df, config)
    logger.info("Slices created: %d", len(slices_df))
    asr_df = asr.run_asr_on_slices(slices_df, config, device=device_str)
    logger.info("ASR rows: %d", len(asr_df))
    emo_df = emotion.run_emotion_on_slices(slices_df, config, device=device_str)
    logger.info("Emotion rows: %d", len(emo_df))
    qual_df = quality.run_quality_on_slices(slices_df, config, device=device_obj, device_tag=device_str)
    logger.info("Quality rows: %d", len(qual_df))

    meta_dir = config.out_root / "metadata"
    segments_full = metadata_io.build_segments_table(meta_dir)
    logger.info("Segments full rows: %d", len(segments_full))
    metadata_io.create_view(segments_full, config)
    logger.info("View generated: %s", config.view_name)

    _ = (asr_df, emo_df, qual_df)
