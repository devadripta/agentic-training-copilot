#!/usr/bin/env python3
"""
Agentic Training Observability System v2.0

Not a log watcher. An intelligent training supervisor.

Architecture:
    Parser → State Machine → Signal Engine → Intelligent Notifier

Key Innovation:
    Narrative-driven state transitions, not metric spam.

Author: AI Training Copilot Project
Version: 2.0 (Production)
"""

import re
import time
import argparse
import requests
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from collections import deque
from pathlib import Path
from enum import Enum, auto
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

# ============== STATE MACHINE ==============

class TrainingPhase(Enum):
    INITIALIZING = auto()
    WARMUP = auto()
    LEARNING_FAST = auto()
    STABLE_CONVERGENCE = auto()
    PLATEAU = auto()
    OVERFITTING_RISK = auto()
    DIVERGING = auto()
    CONVERGED = auto()

    def description(self) -> str:
        descriptions = {
            TrainingPhase.INITIALIZING: "Model warming up, initial gradients flowing",
            TrainingPhase.WARMUP: "Learning rate scheduling, unstable phase",
            TrainingPhase.LEARNING_FAST: "Rapid learning, major patterns being acquired",
            TrainingPhase.STABLE_CONVERGENCE: "Steady improvement, fine-tuning patterns",
            TrainingPhase.PLATEAU: "Convergence slowing, diminishing returns",
            TrainingPhase.OVERFITTING_RISK: "Memorizing training data, validation suffering",
            TrainingPhase.DIVERGING: "Numerical instability or LR too high",
            TrainingPhase.CONVERGED: "Training complete, optimal weights found",
        }
        return descriptions.get(self, "Unknown state")

@dataclass
class StateTransition:
    from_phase: TrainingPhase
    to_phase: TrainingPhase
    timestamp: datetime
    reason: str
    metrics_snapshot: Dict[str, float]

class TrainingStateMachine:
    """Tracks training lifecycle through discrete narrative states"""

    def __init__(self):
        self.current_phase = TrainingPhase.INITIALIZING
        self.transition_history: List[StateTransition] = []
        self.phase_entry_time = datetime.now()
        self.iterations_in_phase = 0
        self.thresholds = {
            'initializing_iters': 20,
            'fast_learning_min_drop': 0.05,
            'plateau_max_improvement': 0.01,
            'overfitting_val_drop': 0.05,
        }

    def update(self, metrics_history: List[TrainingMetrics], 
               val_history: List[ValidationMetrics]) -> Optional[StateTransition]:
        if len(metrics_history) < 10:
            return None

        self.iterations_in_phase += 1
        new_phase = self._determine_phase(metrics_history, val_history)

        if new_phase != self.current_phase:
            transition = StateTransition(
                from_phase=self.current_phase,
                to_phase=new_phase,
                timestamp=datetime.now(),
                reason=self._generate_reason(new_phase, metrics_history, val_history),
                metrics_snapshot=self._snapshot(metrics_history, val_history)
            )
            self.transition_history.append(transition)
            self.current_phase = new_phase
            self.phase_entry_time = datetime.now()
            self.iterations_in_phase = 0
            return transition
        return None

    def _determine_phase(self, metrics: List[TrainingMetrics], 
                        val_metrics: List[ValidationMetrics]) -> TrainingPhase:
        losses = [m.loss for m in metrics]
        current_loss = losses[-1]

        if math.isnan(current_loss) or current_loss > 100:
            return TrainingPhase.DIVERGING

        if len(metrics) < self.thresholds['initializing_iters']:
            return TrainingPhase.INITIALIZING

        if len(val_metrics) >= 2:
            val_trend = self._compute_val_trend(val_metrics)
            train_trend = self._compute_train_trend(metrics)
            if train_trend < -0.01 and val_trend < -0.05:
                return TrainingPhase.OVERFITTING_RISK

        velocity = self._compute_velocity(losses)

        if self.current_phase == TrainingPhase.INITIALIZING:
            return TrainingPhase.LEARNING_FAST if velocity > self.thresholds['fast_learning_min_drop'] else TrainingPhase.WARMUP

        if velocity > self.thresholds['fast_learning_min_drop']:
            return TrainingPhase.LEARNING_FAST
        elif velocity > self.thresholds['plateau_max_improvement']:
            return TrainingPhase.STABLE_CONVERGENCE
        else:
            return TrainingPhase.PLATEAU

    def _compute_velocity(self, losses: List[float], window: int = 50) -> float:
        if len(losses) < window * 2:
            return 0.0
        recent = sum(losses[-window:]) / window
        older = sum(losses[-window*2:-window]) / window
        if older == 0:
            return 0.0
        return (recent - older) / older

    def _compute_val_trend(self, val_metrics: List[ValidationMetrics]) -> float:
        if len(val_metrics) < 2:
            return 0.0
        recent = val_metrics[-1].bbox_mAP
        older = val_metrics[-2].bbox_mAP
        return (recent - older) / older if older > 0 else 0.0

    def _compute_train_trend(self, metrics: List[TrainingMetrics]) -> float:
        if len(metrics) < 100:
            return 0.0
        recent = sum(m.loss for m in metrics[-50:]) / 50
        older = sum(m.loss for m in metrics[-100:-50]) / 50
        return (recent - older) / older if older > 0 else 0.0

    def _generate_reason(self, new_phase: TrainingPhase, 
                        metrics: List[TrainingMetrics],
                        val_metrics: List[ValidationMetrics]) -> str:
        if new_phase == TrainingPhase.PLATEAU:
            velocity = abs(self._compute_velocity([m.loss for m in metrics]))
            return f"Learning velocity dropped to {velocity*100:.1f}% improvement per epoch"
        elif new_phase == TrainingPhase.OVERFITTING_RISK:
            if val_metrics:
                drop = (val_metrics[-2].bbox_mAP - val_metrics[-1].bbox_mAP) / val_metrics[-2].bbox_mAP
                return f"Validation mAP degraded {drop*100:.1f}% while training loss improving"
        elif new_phase == TrainingPhase.LEARNING_FAST:
            return "High gradient flow, rapid pattern acquisition detected"
        elif new_phase == TrainingPhase.DIVERGING:
            return f"Loss exploded to {metrics[-1].loss:.2f}"
        return f"Transitioned from {self.current_phase.name}"

    def _snapshot(self, metrics: List[TrainingMetrics], 
                 val_metrics: List[ValidationMetrics]) -> Dict[str, float]:
        snapshot = {
            'loss': metrics[-1].loss if metrics else 0.0,
            'lr': metrics[-1].lr if metrics else 0.0,
            'acc': metrics[-1].acc if metrics else 0.0,
        }
        if val_metrics:
            snapshot['val_map'] = val_metrics[-1].bbox_mAP
        return snapshot

# ============== ETA PREDICTOR ==============

class ETAPredictor:
    """Predicts training completion with confidence intervals"""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.ema_time_per_epoch: Optional[float] = None
        self.history: List[float] = []

    def record_epoch(self, epoch: int, iterations: int, time_per_iter: float):
        epoch_duration = iterations * time_per_iter
        self.history.append(epoch_duration)
        if self.ema_time_per_epoch is None:
            self.ema_time_per_epoch = epoch_duration
        else:
            self.ema_time_per_epoch = self.alpha * epoch_duration + (1 - self.alpha) * self.ema_time_per_epoch

    def predict(self, current_epoch: int, total_epochs: int, 
                phase: TrainingPhase) -> Dict[str, any]:
        if self.ema_time_per_epoch is None or current_epoch >= total_epochs:
            return {'eta_seconds': 0, 'eta_formatted': 'Complete', 'confidence': 'High', 
                   'confidence_reason': 'Training finished', 'earliest_possible': 'Complete','latest_possible': 'Complete'}

        remaining_epochs = total_epochs - current_epoch

        # Phase-based adjustments
        if phase == TrainingPhase.PLATEAU:
            remaining_epochs = max(2, int(remaining_epochs * 0.5))
            confidence = "Medium"
            reason = "Plateau detected - may converge early"
        elif phase == TrainingPhase.OVERFITTING_RISK:
            remaining_epochs = min(3, remaining_epochs)
            confidence = "High"
            reason = "Overfitting - recommend stopping soon"
        elif phase == TrainingPhase.DIVERGING:
            confidence = "Low"
            reason = "Training unstable"
        elif len(self.history) < 3:
            confidence = "Low"
            reason = "Insufficient data"
        elif len(self.history) < 5:
            confidence = "Medium"
            reason = "Limited sample size"
        else:
            variance = self._compute_variance()
            if variance < 0.1:
                confidence = "High"
                reason = "Stable epoch duration"
            elif variance < 0.25:
                confidence = "Medium"
                reason = "Moderate variance"
            else:
                confidence = "Low"
                reason = "High variance"

        eta_seconds = remaining_epochs * self.ema_time_per_epoch

        if len(self.history) >= 3:
            fastest = min(self.history)
            slowest = max(self.history)
            earliest = remaining_epochs * fastest
            latest = remaining_epochs * slowest
        else:
            earliest = eta_seconds * 0.8
            latest = eta_seconds * 1.2

        return {
            'eta_seconds': int(eta_seconds),
            'eta_formatted': self._format_duration(eta_seconds),
            'confidence': confidence,
            'confidence_reason': reason,
            'remaining_epochs': remaining_epochs,
            'time_per_epoch_avg': self._format_duration(self.ema_time_per_epoch),
            'earliest_possible': self._format_duration(earliest),
            'latest_possible': self._format_duration(latest),
        }

    def _compute_variance(self) -> float:
        if len(self.history) < 2:
            return 0.0
        mean = sum(self.history) / len(self.history)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        return (variance ** 0.5) / mean

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m"
        else:
            return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"

# ============== NOTIFICATION SYSTEM ==============

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

    def send(self, text: str):
        if not self.enabled:
            print(f"\n[NOTIFICATION]\n{text}\n")
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
            print(f"Failed to send: {e}")

class IntelligentNotifier:
    """Generates narrative-driven, decision-focused notifications"""

    def __init__(self, telegram_token: str, chat_id: str):
        self.telegram = TelegramNotifier(telegram_token, chat_id)

    def format_state_transition(self, transition: StateTransition, 
                               eta_info: Dict, state: TrainingState) -> str:
        emoji_map = {
            TrainingPhase.INITIALIZING: "🟡", TrainingPhase.WARMUP: "🟡",
            TrainingPhase.LEARNING_FAST: "🟢", TrainingPhase.STABLE_CONVERGENCE: "🔵",
            TrainingPhase.PLATEAU: "🟠", TrainingPhase.OVERFITTING_RISK: "🔴",
            TrainingPhase.DIVERGING: "💥", TrainingPhase.CONVERGED: "✅"
        }
        emoji = emoji_map.get(transition.to_phase, "ℹ️")

        lines = [
            f"{emoji} *Phase Transition: {transition.to_phase.name.replace('_', ' ')}*",
            f"",
            f"📊 *{state.experiment_name}*",
            f"",
            f"🔄 *Change:* {transition.from_phase.name.replace('_', ' ')} → {transition.to_phase.name.replace('_', ' ')}",
            f"📝 *Reason:* {transition.reason}",
            f"",
            f"📉 *Current State:*",
            f"   Loss: `{transition.metrics_snapshot.get('loss', 0):.4f}`",
            f"   LR: `{transition.metrics_snapshot.get('lr', 0):.2e}`",
        ]

        if 'val_map' in transition.metrics_snapshot:
            lines.append(f"   Val mAP: `{transition.metrics_snapshot['val_map']:.3f}`")

        lines.extend([
            f"",
            f"⏱️ *ETA Prediction:*",
            f"   Remaining: `{eta_info['eta_formatted']}` (*{eta_info['confidence']}*)",
            f"   Range: `{eta_info['earliest_possible']}` - `{eta_info['latest_possible']}`",
            f"   Reason: _{eta_info['confidence_reason']}_"
        ])

        advice = self._get_phase_advice(transition.to_phase)
        if advice:
            lines.extend([f"", f"💡 *Recommendation:* _{advice}_"])

        return "\n".join(lines)

    def _get_phase_advice(self, phase: TrainingPhase) -> str:
        advice_map = {
            TrainingPhase.PLATEAU: "Learning slowed. Consider: (1) Reduce LR 10x, (2) Early stop if val stable, (3) More data?",
            TrainingPhase.OVERFITTING_RISK: "CRITICAL: Stop within 2-3 epochs or increase regularization NOW.",
            TrainingPhase.LEARNING_FAST: "Good progress. Monitor for divergence. No action needed.",
            TrainingPhase.DIVERGING: "KILL TRAINING. Reduce LR 10x and restart from checkpoint.",
            TrainingPhase.STABLE_CONVERGENCE: "Healthy. Continue. Expected 0.5-2% further gains."
        }
        return advice_map.get(phase, "")

    def format_early_termination(self, state: TrainingState, 
                                 epochs_stalled: int, savings: str) -> str:
        if not state.validation_history:
            return ""
        best_val = max(state.validation_history, key=lambda x: x.bbox_mAP)
        current_val = state.validation_history[-1]

        return f"""💰 *Early Termination Recommendation*

📊 *{state.experiment_name}*

⚠️ *Analysis:*
   No improvement for `{epochs_stalled}` epochs
   Best: `{best_val.bbox_mAP:.3f}` (Epoch {best_val.epoch})
   Current: `{current_val.bbox_mAP:.3f}` (Epoch {current_val.epoch})

💡 *Recommendation:* Stop now. P(>0.5% gain) ≈ LOW

💵 *Savings:* ~{savings} GPU hours

🏆 *Use checkpoint:* epoch_{best_val.epoch}.pth"""

    def send_transition(self, transition: StateTransition, eta: Dict, state: TrainingState):
        self.telegram.send(self.format_state_transition(transition, eta, state))

    def send_eta_update(self, state: TrainingState, eta: Dict, phase: TrainingPhase):
        if not state.metrics_history:
            return
        latest = state.metrics_history[-1]
        msg = f"""⏱️ *Progress: {state.experiment_name}*

📊 Epoch {latest.epoch} | {phase.name.replace('_', ' ')}
📉 Loss: `{latest.loss:.4f}` | Acc: `{latest.acc:.1f}%`

⏱️ ETA: `{eta['eta_formatted']}` ({eta['confidence']})
   Range: `{eta['earliest_possible']}` - `{eta['latest_possible']}`

💾 GPU: `{latest.memory}MB`"""
        self.telegram.send(msg)

# ============== LOG PARSER ==============

class MMEngineLogParser:
    def __init__(self):
        self.state = None
        self.current_epoch = 0

    def init_experiment(self, exp_name: str):
        self.state = TrainingState(experiment_name=exp_name)

    def parse_line(self, line: str) -> Optional[Dict]:
        line = line.strip()

        if 'Exp name:' in line and not self.state:
            match = re.search(r'Exp name:\s+(\S+)', line)
            if match:
                self.init_experiment(match.group(1))
                return {'type': 'experiment', 'name': match.group(1)}

        if 'Saving checkpoint' in line:
            match = re.search(r'Saving checkpoint at (\d+) epochs', line)
            if match and self.state:
                self.state.current_epoch = int(match.group(1))
                return {'type': 'checkpoint', 'epoch': int(match.group(1))}

        if any(x in line for x in ['Error', 'Exception', 'Traceback', 'Killed', 'OOM']):
            if self.state:
                self.state.status = "crashed"
            return {'type': 'crash', 'line': line[:200]}

        if 'Epoch(train)' in line:
            return self._parse_train(line)

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
        except:
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

# ============== MAIN COPILOT ==============

class AgenticTrainingCopilot:
    """
    Intelligent training observability system.

    Not a log watcher. A training supervisor that understands context,
    predicts outcomes, and makes recommendations.
    """

    def __init__(self, telegram_token: str = "", telegram_chat: str = ""):
        self.parser = MMEngineLogParser()
        self.state_machine = TrainingStateMachine()
        self.eta_predictor = ETAPredictor()
        self.notifier = IntelligentNotifier(telegram_token, telegram_chat)

        self._last_notification = 0
        self._alert_cooldown = 600
        self._alert_history = {}
        self._epochs_without_improvement = 0
        self._best_val_map = 0.0

    def process_line(self, line: str):
        result = self.parser.parse_line(line)
        if not result or not self.parser.state:
            return

        state = self.parser.state

        # Record epoch timing for ETA
        if result['type'] == 'train' and result['metric'].iteration == result['metric'].total_iterations:
            m = result['metric']
            self.eta_predictor.record_epoch(m.epoch, m.total_iterations, m.time_per_iter)

        # Update state machine
        if result['type'] == 'train' and len(state.metrics_history) % 20 == 0:
            transition = self.state_machine.update(state.metrics_history, state.validation_history)

            if transition:
                eta = self.eta_predictor.predict(
                    state.current_epoch, 
                    state.current_epoch + 5,  # Estimate 5 more epochs
                    self.state_machine.current_phase
                )
                self.notifier.send_transition(transition, eta, state)
                print(f"🔄 State transition: {transition.from_phase.name} → {transition.to_phase.name}")

        # Check for early termination opportunity
        if result['type'] == 'validation':
            self._check_early_termination(state)

            # Periodic ETA update
            eta = self.eta_predictor.predict(
                state.current_epoch, state.current_epoch + 5,
                self.state_machine.current_phase
            )
            self.notifier.send_eta_update(state, eta, self.state_machine.current_phase)

    def _check_early_termination(self, state: TrainingState):
        """Detect if training should stop early to save compute"""
        if len(state.validation_history) < 2:
            return

        current_val = state.validation_history[-1].bbox_mAP

        if current_val > self._best_val_map:
            self._best_val_map = current_val
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        # If 4+ epochs without improvement and in plateau
        if (self._epochs_without_improvement >= 4 and 
            self.state_machine.current_phase == TrainingPhase.PLATEAU):

            # Estimate savings
            avg_epoch_time = self.eta_predictor.ema_time_per_epoch or 1800  # Default 30min
            remaining = 5  # Assume 5 epochs left
            savings_seconds = remaining * avg_epoch_time
            savings_str = self.eta_predictor._format_duration(savings_seconds)

            msg = self.notifier.format_early_termination(
                state, self._epochs_without_improvement, savings_str
            )
            self.notifier.telegram.send(msg)
            print(f"💰 Early termination recommended: {self._epochs_without_improvement} epochs stalled")

    def watch(self, filepath: str, start_end=True):
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
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.get_status()} | Lines: {lines}")
                            last_status = time.time()
                    else:
                        time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped. Processed {lines} lines.")
            self._send_final_report()

    def get_status(self) -> str:
        state = self.parser.state
        if not state or not state.metrics_history:
            return "Waiting..."
        latest = state.metrics_history[-1]
        phase = self.state_machine.current_phase.name[:10]
        return f"E{latest.epoch} {phase} Loss{latest.loss:.3f}"

    def _send_final_report(self):
        state = self.parser.state
        if not state:
            return

        best_val = max(state.validation_history, key=lambda x: x.bbox_mAP) if state.validation_history else None

        msg = f"""✅ *Training Complete: {state.experiment_name}*

📊 Epochs: `{state.current_epoch}` | Phase: `{self.state_machine.current_phase.name}`

🏆 *Best Result:*
   Epoch {best_val.epoch if best_val else 'N/A'} | mAP: `{best_val.bbox_mAP:.3f if best_val else 0}`

🔄 *State Transitions:*
   {len(self.state_machine.transition_history)} phase changes detected

💰 *Compute Saved:*
   Early termination alerts: {sum(1 for a in state.alerts if 'early' in str(a).lower())}

🎉 Training supervised successfully."""

        self.notifier.telegram.send(msg)
        print("\n" + msg.replace('*', '').replace('`', ''))

def main():
    parser = argparse.ArgumentParser(description='Agentic Training Observability System')
    parser.add_argument('--log', required=True, help='Path to training log')
    parser.add_argument('--token', default='', help='Telegram bot token')
    parser.add_argument('--chat', default='', help='Telegram chat ID')
    parser.add_argument('--from-start', action='store_true', help='Read from beginning')

    args = parser.parse_args()

    copilot = AgenticTrainingCopilot(args.token, args.chat)
    copilot.watch(args.log, start_end=not args.from_start)

if __name__ == "__main__":
    main()