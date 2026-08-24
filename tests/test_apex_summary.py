import numpy as np

from src.apex.modules.apex_phase_spotter_roi import ApexPhaseSpotterROI


def _bare_spotter() -> ApexPhaseSpotterROI:
    spotter = object.__new__(ApexPhaseSpotterROI)
    return spotter


def test_summarize_returns_smoothed_and_phases_keys():
    spotter = _bare_spotter()
    rng = np.random.default_rng(42)
    signal = rng.random(50).tolist()

    result = spotter.summarize_signal(signal)

    assert set(result) == {"smoothed_magnitudes", "detected_phases"}
    assert len(result["smoothed_magnitudes"]) == len(signal)
    for phase in result["detected_phases"]:
        assert set(phase) == {"onset", "apex", "offset"}


def test_summarize_falls_back_to_raw_on_failure():
    spotter = _bare_spotter()
    result = spotter.summarize_signal([0.5])
    assert result["smoothed_magnitudes"] == [0.5]
    assert result["detected_phases"] == []
