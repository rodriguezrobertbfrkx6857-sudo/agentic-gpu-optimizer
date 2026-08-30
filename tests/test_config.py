from pathlib import Path

from optimizer.config import load_config


def test_load_transpose_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "transpose_case_study.yaml")
    assert config.name == "transpose_case_study"
    assert config.input_size == 192
    assert config.candidate.variant == "transpose_tiled_padded"
    assert config.gate.min_improvement == 0.03

