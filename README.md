# Agentic GPU Optimizer：AI 辅助 GPU Kernel 优化闭环

一个以基准为驱动的 AI 辅助 GPU Kernel 优化工作流。系统把“优化建议”视为可证伪假设，依次经过构建、正确性、基准、可选 profiler 发现和明确决策门禁；`Manual Provider` 让闭环无需 API key 也能复现，`Command Provider` 则允许在用户明确配置 `GPU_OPTIMIZER_AGENT_COMMAND` 后接入外部 coding-agent CLI。

当前案例使用 CPU fallback，因为本机没有 NVIDIA 驱动和 CUDA Toolkit。每份报告都区分 `cpu_only` 与 CUDA 执行，CUDA 源码仍保留在各 workload 目录中。

## 工作流

```text
工作负载 + 测试/基准规格
          ↓
构建 → 正确性 → 基准 → profiler 发现
          ↓
生成假设 → 校验/应用统一 diff → 候选构建 → 正确性 → 基准
          ↓
             ACCEPT / REJECT / INCONCLUSIVE
```

默认接受条件是正确性通过、中位时间至少改善 3%、变异系数可接受。正确但回退的候选会被拒绝；正确但不稳定或未达到阈值的候选标记为 `INCONCLUSIVE`，阈值位于 YAML 配置中。

## 快速开始

```powershell
python -m pytest
python scripts/run_case_studies.py
```

运行单个配置：

```powershell
python -m optimizer.orchestrator examples/transpose_case_study.yaml
```

每次运行会在 `runs/YYYYMMDD_HHMMSS/` 下写入请求的证据；面向公开展示的摘要位于 [`experiments/transpose_case_study`](experiments/transpose_case_study) 和 [`experiments/reduction_case_study`](experiments/reduction_case_study)。候选代码只有在全新的运行目录、路径校验和 `git apply --check` 通过后才会应用。

## 案例与证据

- 转置：标量 CPU 参考与模拟合并访问的 32×33 CUDA 转置的 NumPy 候选。正确性门禁先于计时，CPU-only 加速比不会被包装成 CUDA 结论。
- 归约：快速 float32 基线与强调精度的 float64 转换候选；当转换成本超过收益时，它提供一个有意义的拒绝案例。

每个实验目录都保存 `environment.json`、`source_hashes.json`、`baseline_build.json`、`candidate_build.json`、`decision.json`、`correctness.json`、`benchmark.json`、`optimization_prompt.md`、`candidate.patch` 和 `candidate_application.json`。运行器记录统一 diff 是否真正应用、`nvcc` 是否构建成功以及时间来自哪个后端；决策不依据候选文本本身。

## Provider 与安全边界

`ManualProvider` 默认不联系模型。`CommandProvider` 只是外部命令适配器，不读取或写入凭据；只有在命令已安装且用户明确想使用时才设置 `GPU_OPTIMIZER_AGENT_COMMAND`。候选仍必须通过同一套门禁。

这是一个工作流演示，不是自动信任代码的机制。AI 生成代码永远不能跳过验证，最终决策以证据门禁为准。

## CUDA 路径与限制

CUDA 主机上，运行器用 `nvcc` 构建基线和候选，用 CUDA Events 计时，并可调用 `ncu` 或 `nsys`。当前 CPU-only 主机执行 Python 参考路径，写入 `NOT BENCHMARKED ON CURRENT HARDWARE`，当 CPU 改善无法证明 CUDA 接受时标记 `INCONCLUSIVE`。不会从 wall-clock 时间推断 profiler counter，也不声称任何未实际运行的 NVIDIA 或 Biren 结果。
