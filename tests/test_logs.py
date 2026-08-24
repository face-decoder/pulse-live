from src.api.logs import _extract_predictions, _PredictionParser

_PREFIX = "2026-08-24 10:00:00,000 - uvicorn - INFO - Sending response to websocket:"


def _prediction_block(label="anxiety_rendah", confidence=0.9):
    return (
        f"{_PREFIX}\n"
        "{\n"
        '  "type": "prediction",\n'
        f'  "label": "{label}",\n'
        f'  "confidence": {confidence},\n'
        '  "latency_ms": 12.5\n'
        "}\n"
    )


def test_parser_extracts_prediction_entry():
    parser = _PredictionParser()
    entry = None
    for line in _prediction_block().splitlines(True):
        result = parser.feed(line)
        if result is not None:
            entry = result
    assert entry is not None
    assert entry["label"] == "anxiety_rendah"
    assert entry["confidence"] == 0.9
    assert entry["latency_ms"] == 12.5


def test_parser_ignores_non_prediction_payloads():
    lines = f'{_PREFIX}\n{{\n  "type": "bbox",\n  "bbox": null\n}}\n'
    parser = _PredictionParser()
    entries = [e for e in (parser.feed(line) for line in lines.splitlines(True)) if e]
    assert entries == []


def test_parser_survives_malformed_json():
    lines = f"{_PREFIX}\n{{broken json\n"
    parser = _PredictionParser()
    assert all(parser.feed(line) is None for line in lines.splitlines(True))
    entry = None
    for line in _prediction_block().splitlines(True):
        result = parser.feed(line)
        if result is not None:
            entry = result
    assert entry is not None


def test_extract_predictions_iterator():
    text = "noise line\n" + _prediction_block() + "more noise\n"
    entries = list(_extract_predictions(text.splitlines(True)))
    assert len(entries) == 1
