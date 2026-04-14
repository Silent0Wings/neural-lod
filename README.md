# Neural LOD: Adaptive Level-of-Detail System

**What it does:** Trains neural networks to predict optimal LOD thresholds for real-time rendering in Unity.

## Overview

Neural LOD optimizes game performance by predicting when to switch between LOD levels based on camera position, velocity, and scene geometry. A trained model runs at runtime in Unity to dynamically adjust LOD thresholds for better performance.

**Current status:** Stages 1-4 complete. Stage 4 adds stability filters to prevent jittery LOD switches.

## Architecture

Three independent subsystems (strict isolation enforced):

### ML Pipeline (`/ml_pipeline`)
- `run_pipeline.py` — Master orchestrator
- `scripts/` — merge, label, evaluate stages
- `data/` — **READ-ONLY** raw CSVs
- `models/` — Trained models for deployment
- `docs/` — Stage breakdown (0-4)

### Unity Integration (`/Assets`)
- `Assets/Script/` — C# LOD applicators
- `Assets/Models/` — Deployed neural models
- `Assets/Scenes/` — Test scenes (Baker, Forest)

### LLM Integration (`/LLM_Integration`)
- Orchestration layer for complex workflows

## Stages

| Stage | What It Does | Key Result |
|-------|--------------|-----------|
| **0** | Verify proposal alignment | All promises delivered ✅ |
| **1-2** | Train LOD predictors | Correct decisions, but jittery |
| **3** | 4-value threshold vectors | Independent LOD level control |
| **4** | Stability filters | Eliminated jitter (500ms dwell + EMA) |

**Stage 4 Fix:** Model made correct decisions (41% win rate) but switched LOD 3x/sec. Solution: Runtime filters without retraining.

## Quick Start

```bash
# Run pipeline
python ml_pipeline/run_pipeline.py

# Specific stage
python ml_pipeline/run_pipeline.py --stage merge
python ml_pipeline/run_pipeline.py --stage label
```

## Rules

- `/ml_pipeline/data/` and `/ProjectSettings/` are **READ-ONLY**
- Strict subsystem isolation (work in one folder at a time)
- All outputs stay within their subsystem
- See `.agents/rules/` for operational guidelines

## Docs

- `ml_pipeline/docs/` — Detailed stage reports (0-4)
- `Unity_Documentation/` — C# implementation
- `.agents/rules/` — Architecture rules

---

**Last Updated:** 2026-04-08  
**Maintainer:** Silent0Wings  
**Status:** Active Development
