# Unity Scripts Analysis - PlantUML Diagrams

Comprehensive PlantUML diagram suite analyzing the Neural LOD Unity scripts across all 4 stages, organized by functional category.

## Diagram Organization

All diagrams are PlantUML `.txt` files organized by stage and category.

### Stage 1: Scalar LOD Bias Prediction

**Path:** `Stage_1/`

| Category | File | Diagram Type | Purpose |
|---|---|---|---|
| **Training** | `training_inference_flow.txt` | Sequence | Model inference with stability controls (hysteresis, dwell, max delta) |
| **Data Collection** | `data_collection_orchestration.txt` | Activity | Orchestrated multi-bias/speed/rotation parameter sweeps |
| **Organisation** | `organisation_data_structures.txt` | Component | Feature extraction pipeline (20D normalized features) |
| **Logging** | `logging_metrics_capture.txt` | Sequence | Frame-by-frame metrics collection into CSV rows |
| **Evaluation** | `evaluation_metrics.txt` | Activity | Performance analysis (latency, stability, efficiency) |

---

### Stage 2: Per-Object Threshold Prediction

**Path:** `Stage_2/`

| Category | File | Diagram Type | Purpose |
|---|---|---|---|
| **Training** | `training_predictor_flow.txt` | Sequence | Threshold predictor inference on baker-generated data |
| **Data Collection** | `data_collection_baker_pipeline.txt` | Activity | 5-step baker pipeline (LOD collect → grid gen → sample → label → export) |
| **Organisation** | `organisation_data_structures.txt` | Class | Data classes (GridPoint, SampleRecord, LabelledSample, ProfilingSession) |
| **Logging** | `logging_baker_metrics.txt` | Sequence | Per-LOD-level metrics collection (~1.7M rows) |
| **Evaluation** | `evaluation_comparison.txt` | Activity | Model vs oracle comparison (MAE, spatial heatmaps, ROI analysis) |

---

### Stage 3: 4-Value Threshold Vectors

**Path:** `Stage_3/`

| Category | File | Diagram Type | Purpose |
|---|---|---|---|
| **Training** | `training_4value_predictor.txt` | Sequence | 4-output MLP with monotonicity enforcement (T0 > T1 > T2 > T3) |
| **Data Collection** | *Inherits from Stage 2* | — | Reuses baker pipeline, applies 4-value relabeling |
| **Organisation** | `organisation_4value_structures.txt` | Component | Threshold vector processing and model architecture |
| **Logging** | `logging_4value_telemetry.txt` | Sequence | Extended telemetry with per-LOD threshold tracking |
| **Evaluation** | `evaluation_4value_analysis.txt` | Activity | Per-LOD MAE, monotonicity, stability, ROI per-level analysis |

---

### Stage 4: REINFORCE RL with Stability Filters

**Path:** `Stage_4/`

| Category | File | Diagram Type | Purpose |
|---|---|---|---|
| **Training** | `training_rl_policy.txt` | Sequence | RL policy inference + stability filters (dwell timer + EMA smoothing) |
| **Data Collection** | `data_collection_rollout.txt` | Activity | Two-phase collection (null rule-based → mixed RL + rule) |
| **Organisation** | `organisation_rl_structures.txt` | Class | RolloutStep, RolloutEpisode, RewardFunction, StabilityFilters |
| **Logging** | `logging_rollout_metrics.txt` | Sequence | RL-specific telemetry (rewards, filter flags, rollout metrics) |
| **Evaluation** | `evaluation_rl_results.txt` | Activity | Policy performance, stability, latency, robustness, stacked improvements |

---

## Diagram Types Used

### By Category

| Category | Preferred Type | Rationale |
|---|---|---|
| **Training** | Sequence | Shows step-by-step inference flow, decision points, filter application |
| **Data Collection** | Activity | Shows orchestrated multi-step pipeline with nested loops, branching |
| **Organisation & Processing** | Component/Class | Shows data structures, transformations, pipeline stages |
| **Logging** | Sequence | Shows frame-by-frame capture, buffering, I/O operations |
| **Evaluation** | Activity | Shows analytical workflow, metric computation, decision trees |

### Summary

- **Sequence Diagrams (8):** Training, Logging across all stages
- **Activity Diagrams (8):** Data Collection, Evaluation across all stages  
- **Class/Component Diagrams (4):** Organisation & Processing for each stage

---

## How to Use These Diagrams

1. **Understand a stage's architecture:**
   - Read Training + Data Collection diagrams first
   - Then review Organisation structures
   - Check Logging pipeline for data output format
   - Study Evaluation criteria for success metrics

2. **Trace data flow:**
   - Start with Data Collection (Activity)
   - Follow to Organisation (structures)
   - See how it feeds into Training (Sequence)
   - Check Logging output format

3. **Understand stability mechanisms:**
   - Stage 1: Hysteresis + Dwell + Max Delta
   - Stage 2: Same + Monotonicity enforcement (implicit)
   - Stage 3: Same + Per-LOD monotonicity (explicit)
   - Stage 4: Dead Zone + Dwell Timer + EMA Smoothing

4. **Compare improvements across stages:**
   - Stage 1 → 2: Spatial awareness (grid-based)
   - Stage 2 → 3: Per-level control (4 independent thresholds)
   - Stage 3 → 4: Adaptive closed-loop (RL policy)

---

## Key Insights by Stage

### Stage 1
- **Focus:** Basic neural LOD bias prediction
- **Data:** Multi-parameter sweeps (bias × speed × rotation)
- **Stability:** Runtime guards (hysteresis, dwell, clamping)
- **Output:** Per-frame bias adjustment

### Stage 2
- **Focus:** Scene-specific baker with spatial awareness
- **Data:** 8K grid points × 216 rotations = 1.7M training samples
- **Stability:** Label monotonicity + runtime enforcement
- **Output:** Single threshold per LOD transition

### Stage 3
- **Focus:** Independent per-LOD-level thresholds
- **Data:** Same baker data, relabeled for 4 outputs
- **Stability:** Strict monotonicity + per-level min gaps
- **Output:** Vector [T0, T1, T2, T3] for independent control

### Stage 4
- **Focus:** Closed-loop RL with reward-driven adaptation
- **Data:** Two-phase rollouts (null baseline → mixed training)
- **Stability:** Dwell (500ms min) + EMA (α=0.2) filters
- **Output:** Bias delta clamped to [-0.20, +0.20]

---

## Testing These Diagrams

To render PlantUML diagrams:

1. **Online:** Visit [PlantUML Online Editor](https://www.plantuml.com/plantuml/uml/)
2. **Local:** Install PlantUML + Graphviz, run:
   ```bash
   plantuml Stage_1/training_inference_flow.txt
   ```
3. **VSCode:** Install PlantUML extension, open `.txt` file, preview

---

## References

- **Scripts Location:** `C:\Users\Gica\neural-lod\Assets\Script\Stage_*`
- **Stage Documentation:** `C:\Users\Gica\neural-lod\ml_pipeline\docs\Stage_*.md`
- **Python Scripts:** `C:\Users\Gica\neural-lod\ml_pipeline\scripts\`, `ml_pipeline\training\`

---

**Generated:** 2026-04-08  
**Purpose:** Visual documentation of Neural LOD Unity architecture across all stages
