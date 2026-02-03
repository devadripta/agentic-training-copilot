#!/usr/bin/env python3
"""
AI Training Copilot - Intelligent observability for deep learning
Watches training logs, detects anomalies, sends Telegram notifications

Setup:
1. Create Telegram bot via @BotFather, get token
2. Get your chat ID via @userinfobot
3. Run: python copilot.py --log /path/to/training.log --token YOUR_TOKEN --chat YOUR_CHAT_ID
"""

import re
import json
import time
import argparse
import requests
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from collections import deque
from pathlib import Path
import math

# ============== DATA STRUCTURES ==============

@dataclass
class TrainingMetrics:
    timestamp: datetime
    epoch: int
    iteration: int
    total_iterations: int
    loss: float
    loss_rpn_cls: float
    loss_rpn_bbox: float
    loss_cls: float
    loss_bbox: float
    loss_mask: float
    acc: float
    lr: float
    memory: int
    time_per_iter: float

@dataclass
class ValidationMetrics:
    timestamp: datetime
    epoch: int
    bbox_mAP: float
    segm_mAP: float
    bbox_mAP_50: float
    bbox_mAP_75: float

@dataclass
class TrainingState:
    experiment_name: str
    current_epoch: int = 0
    total_epochs: int = 0
    status: str = "idle"
    metrics_history: List[TrainingMetrics] = None
    validation_history: List[ValidationMetrics] = None
    alerts: List[Dict] = None

    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []
        if self.validation_history is None:
            self.validation_history = []
        if self.alerts is None:
            self.alerts = []

# ============== SIGNAL ENGINE ==============

class SignalEngine:
    """Detects training pathologies using deterministic analysis"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size

    def detect_convergence_plateau(self, losses: List[float], threshold: float = 0.001) -> Dict:
        """Detect if loss stopped improving"""
        if len(losses) < 20:
            return {"detected": False}

        recent = losses[-10:]
        older = losses[-20:-10] if len(losses) >= 20 else losses[:10]

        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)

        improvement = (older_mean - recent_mean) / older_mean if older_mean > 0 else 0

        if improvement < threshold:
            return {
                "detected": True,
                "severity": "medium",
                "type": "plateau",
                "improvement_rate": f"{improvement*100:.2f}%",
                "message": f"Loss plateau: only {improvement*100:.1f}% improvement in last 10 steps. Consider LR decay."
            }
        return {"detected": False}

    def detect_divergence(self, losses: List[float]) -> Dict:
        """Detect exploding/vanishing loss"""
        if len(losses) < 3:
            return {"detected": False}

        current = losses[-1]

        if math.isnan(current) or math.isinf(current):
            return {
                "detected": True,
                "severity": "critical",
                "type": "nan",
                "message": "🚨 CRITICAL: Loss is NaN/Inf! Training diverged. Check LR and gradients."
            }

        if current > 100:  # Suspiciously high
            return {
                "detected": True,
                "severity": "high",
                "type": "explosion",
                "message": f"⚠️ Loss explosion detected: {current:.2f}. Reduce learning rate immediately."
            }

        return {"detected": False}

    def detect_overfitting(self, train_losses: List[float], val_metrics: List[ValidationMetrics]) -> Dict:
        """Detect overfitting via train/val gap"""
        if len(val_metrics) < 2 or len(train_losses) < 100:
            return {"detected": False}

        recent_val = val_metrics[-1].bbox_mAP
        previous_val = val_metrics[-2].bbox_mAP

        # Check if val dropping while training continues
        if recent_val < previous_val * 0.9:  # 10% drop
            return {
                "detected": True,
                "severity": "high",
                "type": "overfitting",
                "drop": f"{(previous_val-recent_val)*100:.1f}%",
                "message": f"⚠️ OVERFITTING: Val mAP dropped {(previous_val-recent_val)*100:.1f}%. Add regularization or stop."
            }
        return {"detected": False}

    def detect_gpu_issues(self, memory_readings: List[int]) -> Dict:
        """Detect GPU problems"""
        if len(memory_readings) < 10:
            return {"detected": False}

        recent = memory_readings[-10:]
        trend = recent[-1] - recent[0]

        if trend > recent[0] * 0.2:  # 20% growth
            return {
                "detected": True,
                "severity": "high",
                "type": "memory_leak",
                "message": f"⚠️ GPU memory leak: +{trend}MB. Restart training soon."
            }

        if recent[-1] > 22000:  # Near OOM (24GB card)
            return {
                "detected": True,
                "severity": "critical",
                "type": "oom_risk",
                "message": f"🚨 OOM RISK: {recent[-1]/24576*100:.1f}% VRAM used! Reduce batch size."
            }

        return {"detected": False}

    def analyze(self, state: TrainingState) -> List[Dict]:
        """Run all detectors"""
        alerts = []

        if len(state.metrics_history) < 10:
            return alerts

        losses = [m.loss for m in state.metrics_history]
        memories = [m.memory for m in state.metrics_history]

        checks = [
            self.detect_divergence(losses),
            self.detect_convergence_plateau(losses),
            self.detect_gpu_issues(memories),
        ]

        if state.validation_history:
            checks.append(self.detect_overfitting(losses, state.validation_history))

        for check in checks:
            if check.get("detected"):
                alerts.append(check)

        return alerts

# ============== LOG PARSER ==============

class MMEngineLogParser:
    """Parses MMEngine training logs"""

    def __init__(self):
        self.state = None
        self.current_epoch = 0

    def init_experiment(self, exp_name: str):
        self.state = TrainingState(experiment_name=exp_name)
        print(f"📊 Tracking: {exp_name}")

    def parse_line(self, line: str) -> Optional[Dict]:
        line = line.strip()

        # Experiment name
        if 'Exp name:' in line:
            exp_match = re.search(r'Exp name:\s+(\S+)', line)
            if exp_match and not self.state:
                self.init_experiment(exp_match.group(1))
                return {'type': 'experiment', 'name': exp_match.group(1)}

        # Checkpoint
        if 'Saving checkpoint' in line:
            chk_match = re.search(r'Saving checkpoint at (\d+) epochs', line)
            if chk_match and self.state:
                self.state.current_epoch = int(chk_match.group(1))
                return {'type': 'checkpoint', 'epoch': int(chk_match.group(1))}

        # Crash detection
        if any(x in line for x in ['Error', 'Exception', 'Traceback', 'Killed', 'OOM', 'CUDA out of memory']):
            if self.state:
                self.state.status = "crashed"
            return {'type': 'crash', 'line': line[:200]}

        # Training metrics
        if 'Epoch(train)' in line:
            return self._parse_train(line)

        # Validation metrics
        if 'Epoch(val)' in line and 'bbox_mAP:' in line:
            return self._parse_val(line)

        return None

    def _parse_train(self, line: str) -> Optional[Dict]:
        try:
            epoch_match = re.search(r'Epoch\(train\)\s+\[(\d+)\]', line)
            epoch = int(epoch_match.group(1)) if epoch_match else self.current_epoch
            self.current_epoch = epoch

            iter_match = re.search(r'\[(\s*\d+/\d+)\]', line)
            if not iter_match:
                return None

            iter_str = iter_match.group(1).strip()
            current_iter, total_iter = iter_str.split('/')

            def get_float(pattern, default=0.0):
                match = re.search(pattern, line)
                return float(match.group(1)) if match else default

            metric = TrainingMetrics(
                timestamp=datetime.now(),
                epoch=epoch,
                iteration=int(current_iter.strip()),
                total_iterations=int(total_iter.strip()),
                loss=get_float(r'loss:\s+([\d.]+)'),
                loss_rpn_cls=get_float(r'loss_rpn_cls:\s+([\d.]+)'),
                loss_rpn_bbox=get_float(r'loss_rpn_bbox:\s+([\d.]+)'),
                loss_cls=get_float(r'loss_cls:\s+([\d.]+)'),
                acc=get_float(r'acc:\s+([\d.]+)'),
                loss_bbox=get_float(r'loss_bbox:\s+([\d.]+)'),
                loss_mask=get_float(r'loss_mask:\s+([\d.]+)'),
                lr=get_float(r'lr:\s+([\deE.-]+)'),
                memory=int(get_float(r'memory:\s+(\d+)', 0)),
                time_per_iter=get_float(r'time:\s+([\d.]+)')
            )

            if self.state:
                self.state.metrics_history.append(metric)
                self.state.status = "running"

            return {'type': 'train', 'metric': metric}

        except Exception as e:
            return None

    def _parse_val(self, line: str) -> Optional[Dict]:
        try:
            epoch_match = re.search(r'Epoch\(val\)\s+\[(\d+)\]', line)
            epoch = int(epoch_match.group(1)) if epoch_match else self.current_epoch

            def get_float(pattern, default=0.0):
                match = re.search(pattern, line)
                return float(match.group(1)) if match else default

            val_metric = ValidationMetrics(
                timestamp=datetime.now(),
                epoch=epoch,
                bbox_mAP=get_float(r'bbox_mAP:\s+([\d.]+)'),
                segm_mAP=get_float(r'segm_mAP:\s+([\d.]+)'),
                bbox_mAP_50=get_float(r'bbox_mAP_50:\s+([\d.]+)'),
                bbox_mAP_75=get_float(r'bbox_mAP_75:\s+([\d.]+)')
            )

            if self.state:
                self.state.validation_history.append(val_metric)

            return {'type': 'validation', 'metric': val_metric}
        except:
            return None

# ============== TELEGRAM NOTIFIER ==============

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

    def format_message(self, state: TrainingState, alerts: List[Dict]) -> str:
        if not state.metrics_history:
            return "🔄 Waiting for training data..."

        latest = state.metrics_history[-1]

        # Status emoji
        if state.status == "crashed":
            emoji = "💥"
        elif state.status == "completed":
            emoji = "✅"
        elif any(a.get('severity') == 'critical' for a in alerts):
            emoji = "🚨"
        elif alerts:
            emoji = "⚠️"
        else:
            emoji = "🟢"

        # Progress bar
        progress = latest.iteration / latest.total_iterations if latest.total_iterations else 0
        bar = "█" * int(20 * progress) + "░" * (20 - int(20 * progress))

        lines = [
            f"{emoji} *{state.experiment_name}*",
            f"",
            f"📊 Epoch {latest.epoch} | {bar} {progress*100:.0f}%",
            f"📉 Loss: `{latest.loss:.4f}` | Acc: `{latest.acc:.1f}%`",
            f"💾 GPU: `{latest.memory}MB` | LR: `{latest.lr:.2e}`",
        ]

        if state.validation_history:
            val = state.validation_history[-1]
            lines.extend([
                f"",
                f"📈 Val mAP: `{val.bbox_mAP:.3f}` (best: `{max(v.bbox_mAP for v in state.validation_history):.3f}`)"
            ])

        if alerts:
            lines.extend([f"", f"⚡ Alerts:"])
            for alert in alerts[:3]:
                lines.append(f"• {alert['message'][:80]}")

        return "\n".join(lines)

    def format_completion(self, state: TrainingState) -> str:
        lines = [
            f"✅ *Training Complete: {state.experiment_name}*",
            f"",
            f"📊 Epochs: `{state.current_epoch}`",
        ]

        if state.validation_history:
            best = max(state.validation_history, key=lambda x: x.bbox_mAP)
            lines.extend([
                f"🏆 Best: Epoch {best.epoch}",
                f"   bbox_mAP: `{best.bbox_mAP:.3f}`",
                f"   segm_mAP: `{best.segm_mAP:.3f}`",
            ])

        if state.alerts:
            lines.append(f"⚠️ Total alerts: {len(state.alerts)}")

        return "\n".join(lines)

    def send(self, text: str):
        if not self.enabled:
            print(f"\n[TELEGRAM]\n{text}\n")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            response = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=10)
            if response.status_code != 200:
                print(f"Telegram error: {response.text}")
        except Exception as e:
            print(f"Failed to send Telegram: {e}")

# ============== MAIN COPILOT ==============

class TrainingCopilot:
    def __init__(self, telegram_token: str = "", telegram_chat: str = ""):
        self.parser = MMEngineLogParser()
        self.signal_engine = SignalEngine()
        self.notifier = TelegramNotifier(telegram_token, telegram_chat)

        self._last_notification = 0
        self._alert_cooldown = 600
        self._alert_history = {}

    def process_line(self, line: str):
        result = self.parser.parse_line(line)
        if not result or not self.parser.state:
            return

        state = self.parser.state

        # Signal detection every 20 steps
        if (result['type'] == 'train' and 
            len(state.metrics_history) % 20 == 0 and 
            len(state.metrics_history) > 20):

            alerts = self.signal_engine.analyze(state)
            fresh = self._filter_alerts(alerts)

            for alert in fresh:
                state.alerts.append({**alert, 'timestamp': datetime.now().isoformat()})
                print(f"🚨 [{alert['severity'].upper()}] {alert['message']}")

                if alert['severity'] == 'critical':
                    self._notify(state, [alert])

        # Notify on validation
        if result['type'] == 'validation':
            print(f"📈 Validation Epoch {result['metric'].epoch}: mAP={result['metric'].bbox_mAP:.3f}")
            self._notify(state, [])

    def _filter_alerts(self, alerts):
        fresh = []
        now = time.time()
        for alert in alerts:
            key = alert.get('message', '')[:50]
            if now - self._alert_history.get(key, 0) > self._alert_cooldown:
                fresh.append(alert)
                self._alert_history[key] = now
        return fresh

    def _notify(self, state, urgent):
        now = time.time()
        if not urgent and now - self._last_notification < 300:  # 5 min throttle
            return
        self._last_notification = now
        msg = self.notifier.format_message(state, urgent)
        self.notifier.send(msg)

    def watch(self, filepath: str, start_end=True):
        """Watch log file in real-time"""
        path = Path(filepath)
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return

        print(f"👁️  Watching: {filepath}")
        print(f"   Press Ctrl+C to stop\n")

        position = path.stat().st_size if start_end else 0
        lines = 0
        last_status = time.time()

        try:
            with open(filepath, 'r') as f:
                f.seek(position)
                while True:
                    line = f.readline()
                    if line:
                        self.process_line(line)
                        lines += 1

                        if time.time() - last_status > 30:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {lines} lines | {self.get_status()}")
                            last_status = time.time()
                    else:
                        time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped. Processed {lines} lines.")
            self.send_final_report()

    def get_status(self) -> str:
        state = self.parser.state
        if not state or not state.metrics_history:
            return "Waiting..."
        latest = state.metrics_history[-1]
        return f"E{latest.epoch} Loss {latest.loss:.3f}"

    def send_final_report(self):
        state = self.parser.state
        if state:
            self.notifier.send(self.notifier.format_completion(state))

# ============== ENTRY POINT ==============

def main():
    parser = argparse.ArgumentParser(description='AI Training Copilot')
    parser.add_argument('--log', required=True, help='Path to training log file')
    parser.add_argument('--token', default='', help='Telegram bot token')
    parser.add_argument('--chat', default='', help='Telegram chat ID')
    parser.add_argument('--from-start', action='store_true', help='Read from beginning (not tail)')

    args = parser.parse_args()

    copilot = TrainingCopilot(args.token, args.chat)
    copilot.watch(args.log, start_end=not args.from_start)

if __name__ == "__main__":
    main()