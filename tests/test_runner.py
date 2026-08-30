from pathlib import Path
import sys

from optimizer.config import load_config
from optimizer.runner.benchmark import run_benchmark
from optimizer.runner.apply_candidate import apply_candidate
from optimizer.runner.build import build_workload
from optimizer.runner.correctness import run_correctness
from optimizer.providers.manual import ManualProvider
from optimizer.providers.base import Candidate
from optimizer.providers.command_provider import CommandProvider


def test_cpu_runner_build_correctness_benchmark_and_real_patch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "transpose_case_study.yaml")
    build = build_workload(config, config.baseline_variant)
    assert build["status"] == "PASS"
    correctness = run_correctness(config, config.baseline_variant)
    assert correctness["correctness_pass"] is True
    benchmark = run_benchmark(config, config.baseline_variant)
    assert benchmark["status"] == "BENCHMARKED_CPU_ONLY"
    assert benchmark["median_ms"] > 0
    run_dir = tmp_path / "run"
    application = apply_candidate(ManualProvider().suggest(config), run_dir, config)
    assert application["status"] == "APPLIED"
    assert application["mode"] == "isolated_unified_diff"
    assert (run_dir / "candidate_workspace" / "candidate.cu").is_file()


def test_patch_outside_workload_contract_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "transpose_case_study.yaml")
    candidate = Candidate(
        name="unsafe",
        variant=config.candidate.variant,
        rationale="test",
        patch="""diff --git a/escape.txt b/escape.txt
--- /dev/null
+++ b/escape.txt
@@ -0,0 +1 @@
+unsafe
diff --git a/candidate.cu b/candidate.cu
--- /dev/null
+++ b/candidate.cu
@@ -0,0 +1 @@
+placeholder
""",
    )
    application = apply_candidate(candidate, tmp_path / "unsafe", config)
    assert application["status"] == "REJECTED"


def test_command_provider_uses_a_local_deterministic_fixture():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "transpose_case_study.yaml")
    provider = CommandProvider(
        f'"{sys.executable}" -c "print(\\\"fixture-output\\\")"'
    )
    assert provider.available
    assert provider.command
    assert provider.suggest(config).patch.strip() == "fixture-output"
