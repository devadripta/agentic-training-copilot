# Agentic Training Observability 

> (⚠️ Early-stage training observability tool under active development.)

> Not a log watcher. An intelligent training supervisor. 
=======
# Agentic Training Observability v2.1

**Not a log watcher. An intelligent training supervisor.**
(Add platform-agnostic log auto-discovery with intelligent filtering and wait mechanism)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Modern deep learning training is **weakly observable**. Engineers stare at TensorBoard or grep logs, manually inferring:

- Is this converging normally?
- Should I stop early?
- When will this finish?
- Is something going wrong?

**Existing tools visualize. None interpret.**

---

## The Solution

An **agentic observability layer** that understands training as a **narrative**, not a stream of numbers.

### Key Innovation: Training State Machine

Instead of metric spam, we track discrete phases:

```
INITIALIZING → LEARNING_FAST → STABLE_CONVERGENCE → PLATEAU → [DONE]
```

When transition occurs:

```
🟠 PLATEAU DETECTED

Training transitioned from STABLE_CONVERGENCE → PLATEAU
Learning velocity dropped to 0.6%/epoch. Validation mAP plateaued at 0.275.

ETA: 1h 20m (but likely unnecessary)

💡 Recommendation: Stop now. P(>0.5% gain) ≈ LOW
💵 Savings: ~4 GPU hours
```

---

## What's New in v2.1

### 🎯 **Platform-Agnostic Auto-Discovery**

The copilot now **automatically finds your training logs** regardless of environment:

```bash
# Just run this anywhere
python copilot.py --auto
```

Works on:
- ✅ Kaggle notebooks
- ✅ Google Colab
- ✅ Local machines
- ✅ Remote SSH servers
- ✅ Cloud VMs
- ✅ Any environment with logs

**Philosophy:** Files are universal. Platforms are temporary.

---

## Core Principles

### 1. **Deterministic Detection, Narrative Explanation**
- **Math detects** (reliable, cheap)
- **Phases explain** (intuitive, actionable)

### 2. **Decision Support, Not Data Dump**
- "Stop training" not "loss = 0.4523"
- "Save 4 GPU hours" not "epoch 14/20"

### 3. **Predictive Intelligence**
- ETA with confidence intervals
- Early termination recommendations
- Convergence forecasting

---

## Features

| Capability | Description |
|------------|-------------|
| **Auto-Discovery** | Finds latest training log across all environments |
| **State Tracking** | 8-phase lifecycle (Initializing → Converged) |
| **ETA Prediction** | EMA-smoothed with confidence intervals |
| **Early Termination** | Recommends stop when gains unlikely |
| **Overfitting Detection** | Train/val divergence analysis |
| **Divergence Alerts** | NaN/Inf detection with immediate notification |
| **GPU Monitoring** | Memory leak and OOM risk detection |
| **Smart Cooldown** | No notification spam (configurable) |

---

## Installation

```bash
# Single file, minimal dependencies
pip install requests

# Download
curl -O https://raw.githubusercontent.com/yourname/agentic-copilot/main/copilot.py

# Or just copy the file
```

---

## Usage

### 🚀 **Quickstart (Recommended)**

```bash
# Auto-discover and watch
python copilot.py --auto
```

This will:
1. Search common directories for training logs
2. Find the most recent log with actual content
3. Attach and start supervising
4. Wait if no log exists yet

### 📍 **Explicit Path**

```bash
python copilot.py --log work_dirs/exp_name/train.log
```

### 📱 **With Telegram Alerts**

```bash
# Get token from @BotFather
# Get chat ID from @userinfobot

python copilot.py --auto \
  --token "123456:ABC-DEF..." \
  --chat "123456789"
```

### ⏳ **Wait for Training to Start**

```bash
# Wait up to 10 minutes for log to appear
python copilot.py --auto --wait 600
```

### 📂 **Custom Search Directory**

```bash
# Search from specific location
python copilot.py --auto --base-dir /path/to/experiments
```

### 📖 **Read from Beginning**

```bash
python copilot.py --auto --from-start
```

---

## Auto-Discovery Logic

The copilot intelligently searches for logs:

### Search Locations (in order):
1. **Current directory** (`.`)
2. Common ML directories:
   - `./work_dirs`
   - `./outputs`
   - `./runs`
   - `./logs`
   - `./experiments`
3. **Platform hints** (optional, non-breaking):
   - `/kaggle/working` (Kaggle)
   - `/content` (Colab)

### Selection Criteria:
- ✅ File size > 1KB (ignore empty logs)
- ✅ Contains training indicators (`epoch`, `loss`, `train`)
- ✅ Most recently modified

### Smart Features:
- **Recursive search** with `**/*.log` patterns
- **Content validation** (quick heuristic)
- **Platform-agnostic** (works everywhere)
- **No hardcoded paths** (uses filesystem)

---

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

💡 Recommendation: Learning slowed. Consider: 
   (1) Reduce LR 10x, (2) Early stop if val stable, (3) More data?
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

---

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
│ INTELLIGENT NOTIFIER   │
│ • Phase transitions    │
│ • ETA predictions      │
│ • Early termination    │
└────────────────────────┘
      ↓
 Telegram/Console
```

---

## Design Decisions

### Why State Machine?
Humans trust **phases** ("converging") more than **metrics** ("loss=0.452"). 

Netflix shows "unstable connection" not "packet loss 12%".

### Why Deterministic Detection?
Production systems cannot rely on probabilistic reasoning for failure detection.

**LLMs explain. Math detects.**

### Why Platform-Agnostic?
Your tool should work **anywhere logs exist**.

Filesystem is universal. Platforms are temporary.

### Why No Web UI?
**Intelligence density > integrations.**

Single file deployability > feature creep.

---

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

---

## Configuration

**No config files.** Environment variables only:

```bash
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT="your_chat_id"
```

Or pass as arguments:

```bash
python copilot.py --auto --token ... --chat ...
```

---

## Real-World Scenarios

### Scenario 1: Kaggle Notebook
```python
# In your notebook
!pip install requests
!curl -O https://link-to-copilot.py

# Start training in background
!python train.py > /dev/null 2>&1 &

# Start copilot (auto-finds log)
!python copilot.py --auto --token $TELEGRAM_TOKEN --chat $TELEGRAM_CHAT
```

### Scenario 2: Remote GPU Server
```bash
# SSH into server
ssh gpu-box

# Navigate to project
cd ~/experiments/my-model

# Start copilot (auto-discovers)
python copilot.py --auto

# Close laptop, go home
# Phone buzzes with updates
```

### Scenario 3: Local Development
```bash
# Just run it
python copilot.py --auto

# Works from any directory
cd ~/projects/model-training
python ~/tools/copilot.py --auto
```

---

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
- ✅ **Platform-agnostic auto-discovery**
- ✅ Single-file deployment

---

## Roadmap

- [ ] Experiment comparison ("Why did run B outperform A?")
- [ ] Predictive convergence (estimate final mAP at 30% training)
- [ ] Multi-framework (PyTorch Lightning, HF Trainer)
- [ ] REST API for programmatic access
- [ ] Slack/Discord adapters (community request)

---

## Contributing

This is a **taste project**. We welcome PRs that increase **intelligence density**, not feature count.

**Good PRs:**
- ✅ Better phase detection algorithms
- ✅ More accurate ETA prediction
- ✅ New training pathologies detected
- ✅ Support for new log formats

**Bad PRs:**
- ❌ Web UI
- ❌ Kubernetes operators
- ❌ Feature bloat

---

## Testing

```bash
# Test auto-discovery
python copilot.py --auto --wait 0  # Should fail gracefully

# Test with sample log
python copilot.py --log tests/sample.log --from-start

# Test Telegram (dry run)
python copilot.py --auto --token test --chat test
```

---

## Troubleshooting

### "No training log found"
```bash
# Check what would be searched
python -c "from copilot import LogDiscovery; d = LogDiscovery(); print(d.search_paths)"

# Use explicit path
python copilot.py --log /path/to/train.log
```

### "Log found but no metrics"
- Check log format matches MMEngine
- Use `--from-start` to read entire log
- File might be too small (< 1KB)

### Telegram not working
- Verify token with: `https://api.telegram.org/bot<TOKEN>/getMe`
- Get chat ID from: `@userinfobot`
- Check firewall/network restrictions

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

Built for ML engineers who are tired of staring at logs.

**Design inspiration:**
- Unix philosophy (do one thing well)
- Infrastructure as code (platform-agnostic)
- Developer experience (intelligent defaults)

---

## Changelog

### v2.1 (2024-02-04)
- ✨ Platform-agnostic auto-discovery
- ✨ Intelligent log searching with content validation
- ✨ Wait mode for delayed training starts
- ✨ Better error messages and suggestions
- 📝 Enhanced documentation

### v2.0 (2024-02-03)
- 🎉 Initial release
- State machine implementation
- ETA prediction
- Early termination detection
- Telegram notifications
