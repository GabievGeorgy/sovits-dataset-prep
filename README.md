# sovits-dataset-prep

A CLI pipeline for turning raw audio/video into GPT-SoVITS-ready datasets with diarization, slicing, ASR, emotion tagging, MOS estimation, and view export.

## Prerequisites
- Ubuntu 20.04/22.04
- Python 3.10+
- GPU with CUDA (recommended for pyannote, GigaAM, quality metric)
- `ffmpeg`, `libsndfile1`
- Hugging Face token (`HF_TOKEN`)
- Quality uses `torchmetrics` NonIntrusiveSpeechQualityAssessment (no external weights needed)

## Install
```bash
git clone <repo_url>
cd sovits-dataset-prep
bash scripts/install_linux.sh
```

Keep your Hugging Face token handy for pyannote/GigaAM access.

## Run
```bash
python sovits_prep.py \
  --input /data/in \
  --out-root /data/out \
  --view-name default_view \
  --hf-token "$HF_TOKEN" \
  --device auto \
  --min-duration-sec 3.0 \
  --max-duration-sec 15.0 \
  --min-words 2 \
  --min-quality-mos 3.5 \
  --drop-multi-speaker
```

Key options:
- `--input`: folder with source audio/video or a single file
- `--out-root`: where outputs go
- `--view-name`: name for the exported view under `views/`
- `--hf-token`: Hugging Face token (or set `HF_TOKEN`)
- `--device`: `auto` (prefer CUDA when available, falls back to CPU with a warning), `cuda` (requires GPU, errors if unavailable), or `cpu`

- Filters: `--min-duration-sec`, `--max-duration-sec`, `--min-words`, `--min-quality-mos`, `--drop-multi-speaker`, `--allowed-speakers`
- Diarization tuning: `--merge-gap-sec`, `--min-speakers`, `--max-speakers`, `--diar-min-speech-sec`, `--diar-min-pause-sec` (default 1.5s)

Notes:
- The Linux install script (`scripts/install_linux.sh`) runs a quick `pytest` suite after installing dependencies to catch obvious breakages early.
- On each run, the output directory (`--out-root`) is cleared before processing to keep runs idempotent; do not point it at your input folder.
- Views include `train.list` with absolute paths and per-speaker files `train_<speaker>.list`.

## Outputs
```
out_root/
  raw_audio/            # normalized 16k mono wavs
  segments/             # sliced wavs grouped by speaker
  metadata/
    sources.csv
    diarization.csv
    slices.csv
    asr.csv
    emotions.csv
    quality.csv
    segments_full.csv
  views/
    <view_name>/
      segments.csv
      train.list        # wav_path|speaker_id|ru|text
```

`segments_full.csv` joins all metadata (slice timing, ASR text, emotion scores, MOS scores, config hashes) and is written with UTF-8-sig for easy Windows consumption. All `.csv` files follow comma-separated, UTF-8-sig encoding.
