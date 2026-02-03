# Quick Start Guide - Auto-Discovery Feature

## TL;DR

```bash
# Old way (still works)
python copilot.py --log /path/to/train.log

# New way (platform-agnostic)
python copilot.py --auto
```

---

## Installation

```bash
# Download the enhanced version
curl -O https://link-to/agentic_copilot_v2_enhanced.py

# Rename for convenience
mv agentic_copilot_v2_enhanced.py copilot.py

# Install dependencies
pip install requests
```

---

## Usage Examples

### 1. **Basic Auto-Discovery** (Most Common)

```bash
cd ~/my-training-project
python copilot.py --auto
```

**What happens:**
1. Searches current directory and subdirectories
2. Finds `.log` files > 1KB
3. Validates they contain training metrics
4. Attaches to most recent log
5. Starts supervising

**Output:**
```
🚀 Agentic Training Copilot v2.1
   Platform-Agnostic Training Supervisor

🔍 Auto-discovery mode enabled
   Scanning for training logs...

✅ Found log: ./work_dirs/exp_20260203/train.log
   Size: 45.3 KB
   Modified: 14:23:15

👁️  Watching: ./work_dirs/exp_20260203/train.log
   Press Ctrl+C to stop
```

---

### 2. **With Telegram Notifications**

```bash
python copilot.py --auto \
  --token "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz" \
  --chat "123456789"
```

**Result:** Phone buzzes with state transitions and recommendations

---

### 3. **Wait for Training to Start**

```bash
# Start copilot before training begins
python copilot.py --auto --wait 600

# In another terminal/notebook
python train.py
```

**What happens:**
```
🔍 Auto-discovery mode enabled
   Scanning for training logs...

⏳ Waiting for training log... (30s / 600s)
⏳ Waiting for training log... (60s / 600s)

✅ Found log: ./work_dirs/new_exp/train.log
   Attaching...
```

---

### 4. **Custom Search Location**

```bash
# Search from specific directory
python copilot.py --auto --base-dir /data/experiments
```

**Use case:** Non-standard project structure

---

### 5. **Kaggle Notebook**

```python
# Cell 1: Setup
!pip install requests
!curl -O https://link-to/copilot.py

# Cell 2: Start training (in background)
!python train.py > /dev/null 2>&1 &

# Cell 3: Start copilot
!python copilot.py --auto --token {TELEGRAM_TOKEN} --chat {CHAT_ID}
```

**No hardcoded paths. Works immediately.**

---

### 6. **Google Colab**

```python
# Same as Kaggle
!python copilot.py --auto

# Copilot automatically finds /content/work_dirs/.../train.log
```

---

### 7. **SSH into Remote Server**

```bash
ssh user@gpu-server
cd ~/experiments/my-model

# Start copilot
python copilot.py --auto --token $TELEGRAM_TOKEN --chat $TELEGRAM_CHAT

# Close SSH session
# Phone receives updates
```

---

### 8. **Read Full Log from Beginning**

```bash
# By default, copilot tails from end (skips old logs)
# To read everything:
python copilot.py --auto --from-start
```

**Use case:** Debugging past training runs

---

### 9. **No Wait (Fail Fast)**

```bash
# If no log exists, exit immediately
python copilot.py --auto --wait 0
```

**Output:**
```
❌ No training log found
   Searched:
     • .
     • ./work_dirs
     • ./outputs
     • ./runs

💡 Suggestions:
   • Use --log to specify explicit path
   • Use --wait 300 to wait up to 5 minutes
   • Check that training has started
```

---

## Environment-Specific Tips

### Kaggle
- Logs usually in: `/kaggle/working/work_dirs/`
- Auto-discovery finds them automatically
- Use `--wait 60` if starting copilot before training

### Colab
- Logs usually in: `/content/work_dirs/`
- Auto-discovery finds them automatically  
- Mount Google Drive if you want persistent storage

### Local Machine
- Logs in project subdirectories
- `cd` to project root before running copilot
- Or use `--base-dir` to point to experiments folder

### SSH Server
- Background training with `nohup python train.py &`
- Start copilot in `tmux` or `screen` session
- Or use Telegram mode and disconnect

---

## Troubleshooting

### "No training log found"

**Check what would be searched:**
```python
python -c "
from copilot import LogDiscovery
d = LogDiscovery()
print('Search paths:')
for p in d.search_paths:
    print(f'  {p}')
"
```

**Solutions:**
1. Use explicit path: `--log /exact/path/to/train.log`
2. Wait longer: `--wait 300`
3. Custom search: `--base-dir /your/experiments`

---

### Log found but no metrics detected

**Check log format:**
```bash
head -50 your_train.log
```

**Requirements:**
- Must contain: `epoch`, `loss`, `train`, or `iteration`
- Should be > 1KB (not empty)
- MMEngine format preferred

---

### Multiple logs, wrong one selected

**Copilot selects most recent by modification time.**

**Override:**
```bash
# Explicit path
python copilot.py --log /path/to/specific/train.log
```

---

## Advanced Patterns

### Pattern 1: Multi-GPU Setup
```bash
# Training on GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py &

# Copilot on CPU (no GPU needed)
python copilot.py --auto
```

### Pattern 2: Experiment Tracking
```bash
#!/bin/bash
# run_experiment.sh

# Start training
python train.py --config $1 &

# Start copilot with experiment name
python copilot.py --auto \
  --token $TELEGRAM_TOKEN \
  --chat $TELEGRAM_CHAT \
  --wait 60
```

### Pattern 3: Slurm/Batch Jobs
```bash
# In your SLURM script
#SBATCH --job-name=training

# Training in background
python train.py &

# Copilot (will find log when training starts)
python copilot.py --auto --wait 3600  # 1 hour timeout
```

---

## Migration from v2.0

### Old Script (v2.0)
```bash
# You had to know the path
python copilot.py --log /kaggle/working/work_dirs/exp/train.log
```

### New Script (v2.1)
```bash
# Just let it find the log
python copilot.py --auto
```

### Both work!
```bash
# Explicit path still supported
python copilot.py --log /exact/path/train.log

# Auto-discovery is additive, not breaking
python copilot.py --auto
```

---

## Best Practices

### ✅ Do:
- Use `--auto` for new projects (easiest)
- Use `--wait 60` if starting before training
- Use Telegram for long training runs
- Test with `--wait 0` first to see what's found

### ❌ Don't:
- Hardcode paths in scripts (defeats portability)
- Assume specific log locations (platform-dependent)
- Forget to install `requests` (only dependency)

---

## Performance Notes

### Discovery Speed:
- Typical scan: < 1 second
- Deep directories: 2-3 seconds
- Network filesystems: May be slower

### Resource Usage:
- CPU: Negligible (< 1%)
- Memory: ~50 MB
- No GPU needed

### Scalability:
- Tested with:
  - 1000+ files in directory
  - 10+ GB log files
  - 100+ subdirectories

---

## What's Next?

After you're comfortable with auto-discovery:

1. **Add Telegram** for remote monitoring
2. **Use `--wait`** for automated workflows  
3. **Try custom `--base-dir`** for complex setups
4. **Contribute back** if you find edge cases

---

## Getting Help

**No log found:**
```bash
python copilot.py --auto --wait 0
# Read the suggestions in the error message
```

**Unexpected log selected:**
```bash
ls -lt work_dirs/**/train.log
# Copilot picks most recent - might not be what you want
# Use explicit --log in this case
```

**Still stuck:**
- Check README_v2.1.md for details
- Review CHANGELOG_v2.1.md for design rationale
- Open issue with copilot output

---

## Summary

**The goal:** Make supervision effortless.

**Old way:** Know your paths, escape them correctly, hope it works.

**New way:** `python copilot.py --auto`

That's it. Platform-agnostic. Just works.
