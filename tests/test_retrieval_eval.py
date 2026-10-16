"""Unit tests for ranking frame hits and evaluating labels."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval_eval import (
    attribute_value,
    label_prevalence,
    precision_at_k,
    unique_clips_from_hits,
)


@pytest.fixture
def data():
    meta = [
        {"clip": "rain"},
        {"clip": "rain"},  # A second frame from the same video.
        {"clip": "clear"},
        {"clip": "snow"},
    ]
    labels = {
        "rain": {"attributes": {"weather": "rainy", "timeofday": "night"}},
        "clear": {"attributes": {"weather": "clear", "timeofday": "night"}},
        "snow": {"attributes": {"weather": "snowy", "timeofday": "day"}},
    }
    return meta, labels


def test_deduplicates_frames_and_preserves_rank_order(data):
    meta, _ = data
    assert unique_clips_from_hits([0, 1, 2], meta) == ["rain", "clear"]
    assert unique_clips_from_hits([2, 3, 0], meta, limit=2) == ["clear", "snow"]


def test_skips_invalid_faiss_result_indices(data):
    meta, _ = data
    assert unique_clips_from_hits([-1, 3, 99], meta) == ["snow"]


def test_precision_uses_returned_clip_count(data):
    _, labels = data
    assert precision_at_k(["rain", "clear", "snow"], labels, "weather", "rainy") == pytest.approx(1 / 3)


def test_baseline_counts_unique_indexed_clips(data):
    _, labels = data
    assert label_prevalence(["rain", "rain", "clear", "snow"], labels, "weather", "rainy") == pytest.approx(1 / 3)


def test_attribute_lookup_explains_missing_annotation(data):
    _, labels = data
    with pytest.raises(KeyError, match="weather"):
        attribute_value(labels, "unknown", "weather")


def test_empty_rankings_have_zero_precision(data):
    _, labels = data
    assert precision_at_k([], labels, "weather", "rainy") == 0.0
