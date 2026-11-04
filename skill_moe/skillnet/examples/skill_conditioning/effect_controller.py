"""Small causal controllers; no simulator dependencies or rollout lookahead."""
from dataclasses import dataclass, field

import numpy as np


def causal_vector(feature, state, previous_state, previous_command, step, cap=800):
    feature = np.asarray(feature, np.float32)
    state = np.asarray(state, np.float32)
    previous = state if previous_state is None else np.asarray(previous_state, np.float32)
    return np.concatenate([feature, state, state - previous,
                           np.asarray(previous_command, np.float32), [step / cap]]).astype(np.float32)


@dataclass
class Belief:
    high: float = .8
    low: float = .2
    stable: int = 2
    state: str = "unknown"
    positive: int = 0
    negative: int = 0
    ever: bool = False

    def update(self, probability):
        self.positive = self.positive + 1 if probability >= self.high else 0
        self.negative = self.negative + 1 if probability <= self.low else 0
        if self.positive >= self.stable:
            self.ever = True
            self.state = "achieved"
        elif self.negative >= self.stable:
            self.state = "regressed" if self.ever else "unmet"
        else:
            self.state = "unknown"
        return self.state


@dataclass
class InstructionController:
    condition: str
    beliefs: list = field(default_factory=lambda: [Belief(), Belief()])
    switched: bool = False
    locked: bool = False
    switch_count: int = 0
    rollback_count: int = 0
    time_step: int = 150

    def decide(self, probabilities, step, effect=None):
        states = [belief.update(float(p)) for belief, p in zip(self.beliefs, probabilities)]
        if self.condition == "original":
            return False, states
        if self.switched and states[0] == "regressed":
            self.switched = False
            self.locked = True
            self.rollback_count += 1
        if self.locked or self.switched:
            return self.switched, states
        ready = states[0] == "achieved"
        if self.condition == "time":
            ready = step >= self.time_step
        if self.condition == "effect":
            ready = ready and effect is not None and effect[0] > .05 and effect[1] > 0
        if ready:
            self.switched = True
            self.switch_count += 1
        return self.switched, states


def predict_probabilities(model, vector):
    x = model["transform"].transform(np.asarray(vector)[None])
    return np.asarray([head.predict_proba(x)[0, 1] if hasattr(head, "predict_proba")
                       else head for head in model["heads"]], float)


def predict_effect(model, vector, probabilities):
    x = model["transform"].transform(np.concatenate([vector, probabilities])[None])
    predictions = np.asarray([head.predict(x)[0] for head in model["heads"]])
    return float(predictions.mean()), float(np.quantile(predictions, .1))
