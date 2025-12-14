import pandas as pd

from sovits_prep.slicing import _merge_segments


def test_merge_gap_does_not_cross_other_speaker() -> None:
    diar_df = pd.DataFrame(
        [
            {"source_id": 1, "speaker_id": "A", "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0},
            {"source_id": 1, "speaker_id": "B", "start_sec": 1.2, "end_sec": 1.8, "duration_sec": 0.6},
            {"source_id": 1, "speaker_id": "A", "start_sec": 2.0, "end_sec": 3.0, "duration_sec": 1.0},
        ]
    )

    merged = _merge_segments(diar_df, gap=2.0)
    a = merged[merged["speaker_id"] == "A"].sort_values("start_sec").reset_index(drop=True)

    assert len(a) == 2
    assert a.loc[0, "start_sec"] == 0.0 and a.loc[0, "end_sec"] == 1.0
    assert a.loc[1, "start_sec"] == 2.0 and a.loc[1, "end_sec"] == 3.0


def test_merge_gap_merges_when_clear_gap() -> None:
    diar_df = pd.DataFrame(
        [
            {"source_id": 1, "speaker_id": "A", "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0},
            {"source_id": 1, "speaker_id": "A", "start_sec": 1.4, "end_sec": 2.0, "duration_sec": 0.6},
            {"source_id": 1, "speaker_id": "B", "start_sec": 3.1, "end_sec": 3.3, "duration_sec": 0.2},
        ]
    )

    merged = _merge_segments(diar_df, gap=0.5)
    a = merged[merged["speaker_id"] == "A"].sort_values("start_sec").reset_index(drop=True)

    assert len(a) == 1
    assert a.loc[0, "start_sec"] == 0.0 and a.loc[0, "end_sec"] == 2.0


def test_merge_gap_merges_overlapping_segments() -> None:
    diar_df = pd.DataFrame(
        [
            {"source_id": 1, "speaker_id": "A", "start_sec": 0.0, "end_sec": 2.0, "duration_sec": 2.0},
            {"source_id": 1, "speaker_id": "A", "start_sec": 1.5, "end_sec": 3.0, "duration_sec": 1.5},
        ]
    )

    merged = _merge_segments(diar_df, gap=0.0)
    a = merged[merged["speaker_id"] == "A"].sort_values("start_sec").reset_index(drop=True)

    assert len(a) == 1
    assert a.loc[0, "start_sec"] == 0.0 and a.loc[0, "end_sec"] == 3.0

