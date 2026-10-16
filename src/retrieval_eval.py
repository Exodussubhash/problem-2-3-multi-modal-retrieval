"""Pure helpers for ranking frame hits and evaluating clip labels."""


def unique_clips_from_hits(indices, meta, limit=10):
    """Return the first occurrence of each clip in ranked frame-hit indices."""
    clips = []
    seen = set()
    for index in indices:
        if index < 0 or index >= len(meta):
            continue
        clip = meta[index]["clip"]
        if clip not in seen:
            seen.add(clip)
            clips.append(clip)
            if len(clips) == limit:
                break
    return clips


def attribute_value(labels, clip, field):
    """Get a clip attribute, raising a useful error if annotation is missing."""
    try:
        return labels[clip]["attributes"][field]
    except KeyError as exc:
        raise KeyError(f"Missing label, attributes, or field {field!r} for clip {clip!r}") from exc


def precision_at_k(clips, labels, field, value):
    """Fraction of retrieved clips with the requested label; empty input is 0."""
    if not clips:
        return 0.0
    hits = sum(attribute_value(labels, clip, field) == value for clip in clips)
    return hits / len(clips)


def label_prevalence(clips, labels, field, value):
    """Fraction of indexed clips with the requested label."""
    clips = set(clips)
    if not clips:
        return 0.0
    hits = sum(attribute_value(labels, clip, field) == value for clip in clips)
    return hits / len(clips)
