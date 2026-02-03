# Agentic Training Observability 

> (⚠️ Early-stage training observability tool under active development.)

> Not a log watcher. An intelligent training supervisor. 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Modern deep learning training is **weakly observable**. Engineers stare at TensorBoard or grep logs, manually inferring:
- Is this converging normally?
- Should I stop early?
- When will this finish?
- Is something going wrong?

Existing tools **visualize**. None **interpret**.

## The Solution

An **agentic observability layer** that understands training as a narrative, not a stream of numbers.

### Key Innovation: Training State Machine

Instead of metric spam, we track discrete phases:

```
INITIALIZING → LEARNING_FAST → STABLE_CONVERGENCE → PLATEAU → [DONE]
```

When transition occurs:

> 🟠 **PLATEAU DETECTED**
> 
> Training transitioned from STABLE_CONVERGENCE → PLATEAU
> 
> Learning velocity dropped to 0.6%/epoch. Validation mAP plateaued at 0.275.
> 
> ⏱️ ETA: 1h 20m (but likely unnecessary)  
> 💡 **Recommendation:** Stop now. P(>0.5% gain) ≈ LOW  
> 💵 **Savings:** ~4 GPU hours

### Core Principles

1. **Deterministic Detection, Narrative Explanation**
   - Math detects (reliable, cheap)
   - Phases explain (intuitive, actionable)

2. **Decision Support, Not Data Dump**
   - "Stop training" not "loss = 0.4523"
   - "Save 4 GPU hours" not "epoch 14/20"

3. **Predictive Intelligence**
   - ETA with confidence intervals
   - Early termination recommendations
   - Convergence forecasting

## Features

| Capability | Description |
|------------|-------------|
| **State Tracking** | 9-phase lifecycle (Initializing → Converged) |
| **ETA Prediction** | EMA-smoothed with confidence intervals |
| **Early Termination** | Recommends stop when gains unlikely |
| **Overfitting Detection** | Train/val divergence analysis |
| **Divergence Alerts** | NaN/Inf detection with immediate notification |
| **GPU Monitoring** | Memory leak and OOM risk detection |
| **Smart Cooldown** | No notification spam (configurable) |

## Installation

```bash
# Single file, no dependencies except requests
pip install requests

# Download
curl -O https://raw.githubusercontent.com/yourname/agentic-copilot/main/agentic_copilot_v2.py

# Run
python agentic_copilot_v2.py --log /path/to/training.log
```

## Usage

### Basic (Console Mode)
```bash
python agentic_copilot_v2.py --log work_dirs/exp_name/train.log
```

### With Telegram Alerts
```bash
# Get token from @BotFather
# Get chat ID from @userinfobot

python agentic_copilot_v2.py \
    --log work_dirs/exp_name/train.log \
    --token "123456:ABC-DEF..." \
    --chat "123456789"
```

### From Start of Log
```bash
python agentic_copilot_v2.py --log train.log --from-start
```

## Example Output

### Phase Transition
```
🟠 PLATEAU DETECTED

📊 config_dent_dcn_final_20260203_104304

🔄 Change: STABLE_CONVERGENCE → PLATEAU
📝 Reason: Learning velocity dropped to 0.6% improvement per epoch

📉 Current State:
   Loss: 0.4523
   LR: 1.00e-05
   Val mAP: 0.275

⏱️ ETA Prediction:
   Remaining: 1h 20m (Medium)
   Range: 1h 5m - 1h 40m
   Reason: Plateau detected - may converge early

💡 Recommendation: Learning slowed. Consider: (1) Reduce LR 10x, 
(2) Early stop if val stable, (3) More data?
```

### Early Termination
```
💰 Early Termination Recommendation

📊 config_dent_dcn_final_20260203_104304

⚠️ Analysis:
   No improvement for 4 epochs
   Best: 0.275 (Epoch 10)
   Current: 0.274 (Epoch 14)

💡 Recommendation: Stop now. P(>0.5% gain) ≈ LOW

💵 Savings: ~2h 40m GPU hours

🏆 Use checkpoint: epoch_10.pth
```

## Architecture

```
MMEngine Logs
    ↓
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   PARSER    │ → │ STATE MACH  │ → │  ETA ENGINE │
│  (Extract)  │   │ (Narrative) │   │  (Predict)  │
└─────────────┘   └─────────────┘   └─────────────┘
                      ↓
         ┌────────────────────────┐
         │  INTELLIGENT NOTIFIER  │
         │  • Phase transitions   │
         │  • ETA predictions     │
         │  • Early termination   │
         └────────────────────────┘
                      ↓
               Telegram/Console
```

### Design Decisions

**Why State Machine?**
> Humans trust phases ("converging") more than metrics ("loss=0.452"). 
> Netflix shows "unstable connection" not "packet loss 12%".

**Why Deterministic Detection?**
> Production systems cannot rely on probabilistic reasoning for failure detection.
> LLMs explain. Math detects.

**Why No Web UI?**
> Intelligence density > integrations. 
> Single file deployability > feature creep.

## State Definitions

| State | Description | Typical Action |
|-------|-------------|----------------|
| **INITIALIZING** | First 20 iterations, unstable | Wait |
| **WARMUP** | LR scheduling active | Wait |
| **LEARNING_FAST** | >5% improvement/epoch | Monitor |
| **STABLE_CONVERGENCE** | 1-5% improvement/epoch | Continue |
| **PLATEAU** | <1% improvement/epoch | Consider stopping |
| **OVERFITTING_RISK** | Val degrading, train improving | Stop soon |
| **DIVERGING** | Loss explosion/NaN | Kill immediately |
| **CONVERGED** | Training complete | Done |

## Configuration

No config files. Environment variables only:

```bash
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT="your_chat_id"
```

Or pass as arguments:
```bash
python agentic_copilot_v2.py --token ... --chat ...
```

## Philosophy

### What We Did NOT Build
- ❌ Web dashboard (intelligence density > UI)
- ❌ Slack integration (Telegram sufficient)
- ❌ Docker/K8s (single file deployability)
- ❌ Multi-framework (master one first)
- ❌ LLM explanations (deterministic detection first)

### What We DID Build
- ✅ State machine narrative
- ✅ Predictive ETA
- ✅ Early termination
- ✅ Compute savings calculation
- ✅ Single-file deployment

## Roadmap

- [ ] Experiment comparison ("Why did run B outperform A?")
- [ ] Predictive convergence (estimate final mAP at 30% training)
- [ ] Multi-framework (PyTorch Lightning, HF Trainer)
- [ ] REST API for programmatic access

## Contributing

This is a **taste** project. We welcome PRs that increase intelligence density, not feature count.

Good PRs:
- Better phase detection algorithms
- More accurate ETA prediction
- New training pathologies detected

Bad PRs:
- Web UI
- Slack/Discord integrations
- Kubernetes operators

## License

MIT License - See LICENSE file

## Acknowledgments

Built for ML engineers who are tired of staring at logs.
