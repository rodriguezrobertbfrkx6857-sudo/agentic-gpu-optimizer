from pathlib import Path

from optimizer.config import load_config
from optimizer.runner.benchmark import run_benchmark
from optimizer.runner.apply_candidate import apply_candidate
from optimizer.runner.build import build_workload
from optimizer.runner.correctness import run_correctness
from optimizer.providers.manual import ManualProvider


def test_cpu_runner_build_correctness_and_benchmark():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "transpose_case_study.yaml")
    build = build_workload(config, config.baseline_variant)
    assert build["status"] == "PASS"
    correctness = run_correctness(config, config.baseline_variant)
    assert correctness["correctness_pass"] is True
    benchmark = run_benchmark(config, config.baseline_variant)
    assert benchmark["status"] == "BENCHMARKED_CPU_ONLY"
    assert benchmark["median_ms"] > 0
    application = apply_candidate(ManualProvider().suggest(config), Path("run"))
    assert application["status"] == "APPLIED"
