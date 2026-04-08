from types import SimpleNamespace
from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


def safe_score(x):
    return max(0.001, min(0.999, x))
try:
    from ..models import AiQueryRoutingAction, AiQueryRoutingObservation
except ImportError:
    from models import AiQueryRoutingAction, AiQueryRoutingObservation

try:
    from tasks.grader import grade_task1, grade_task2, grade_task3
except ImportError:
    from grader import grade_task1, grade_task2, grade_task3


AVAILABLE_MODELS = [
    {"id": "llama",        "tier": "small",  "cost": 0.00, "quality": 0.78},
    {"id": "gemini-flash", "tier": "small",  "cost": 0.05, "quality": 0.84},
    {"id": "claude",       "tier": "medium", "cost": 0.10, "quality": 0.94},
    {"id": "gpt4",         "tier": "large",  "cost": 0.20, "quality": 0.96},
]

QUERIES = [
    ("Explain refund policy",          "factual",   0.2),
    ("Solve calculus problem",         "reasoning", 0.8),
    ("Write a Python API",             "code",      0.7),
    ("Summarize legal document",       "reasoning", 0.6),
    ("Translate text to French",       "factual",   0.3),
    ("Debug this SQL query",           "code",      0.65),
    ("What is the capital of France",  "factual",   0.1),
    ("Explain quantum entanglement",   "reasoning", 0.9),
]

MAX_STEPS     = 10
EPISODE_BUDGET = 1.0


class AiQueryRoutingEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.accuracy_history = []
        self.query = ""
        self.query_domain = "factual"
        self.complexity = 0.5
        self.budget = EPISODE_BUDGET
        self.total_cost = 0.0
        self.resolved = False
        self.total_tokens = 0

    def reset(self, seed=None) -> AiQueryRoutingObservation:
        if seed is not None:
            random.seed(seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.accuracy_history = []
        self.total_cost = 0.0
        self.resolved = False
        self.total_tokens = 0

        query_text, domain, complexity = random.choice(QUERIES)
        self.query = query_text
        self.query_domain = domain
        self.complexity = complexity
        self.budget = EPISODE_BUDGET

        return AiQueryRoutingObservation(
            query=self.query,
            query_domain=self.query_domain,
            complexity=self.complexity,
            budget_remaining=self.budget,
            accuracy=0.001,
            last_accuracy=0.001,
            step_number=0,
            accuracy_history=[],
            available_models=AVAILABLE_MODELS,
            reward=0.001,
            done=False,
        )

    def step(self, action: AiQueryRoutingAction) -> AiQueryRoutingObservation:
        self._state.step_count += 1

        model_id = getattr(action, "model_id", "gemini-flash")
        model_tier = getattr(action, "model_tier", "small")

        model_info = next(
            (m for m in AVAILABLE_MODELS if m["id"] == model_id), None
        )

        if model_info:
            base_quality = model_info["quality"]
            cost = model_info["cost"]
        else:
            tier_defaults = {
                "small":  {"quality": 0.75, "cost": 0.05},
                "medium": {"quality": 0.85, "cost": 0.10},
                "large":  {"quality": 0.95, "cost": 0.20},
            }
            d = tier_defaults.get(model_tier, tier_defaults["small"])
            base_quality = d["quality"]
            cost = d["cost"]

        # accuracy drops with complexity
        accuracy = base_quality - (self.complexity * 0.2)
        accuracy = round(max(0.001, min(0.999, accuracy)), 4)

        self.budget = round(max(0.0, self.budget - cost), 4)
        self.total_cost = round(self.total_cost + cost, 4)
        self.total_tokens += int(getattr(action, "max_tokens", 512))
        self.accuracy_history.append(accuracy)

        # resolution bonus for task 3
        resolution_bonus = 0.0
        if self.task_id == 3 and self._state.step_count >= 3:
            self.resolved = True
            resolution_bonus = 0.2

        # quality bonus if accuracy beats previous mean
        quality_bonus = 0.0
        if len(self.accuracy_history) > 1:
            prev_mean = sum(self.accuracy_history[:-1]) / len(self.accuracy_history[:-1])
            if accuracy > prev_mean:
                quality_bonus = 0.1

        tokens_estimate = int(getattr(action, "max_tokens", 512))
        reward = (
            accuracy
            - (tokens_estimate * 0.001)
            + resolution_bonus
            + quality_bonus
        )
        reward = round(max(0.001, min(0.999, reward)), 4)

        if self.task_id == 1:
            done = self._state.step_count >= 1
        elif self.task_id == 2:
            done = self._state.step_count >= 5 or self.budget <= 0.001
        else:  # task 3
            done = self._state.step_count >= MAX_STEPS or self.budget <= 0.001

        if done:
            self.resolved = True

        return AiQueryRoutingObservation(
            query=self.query,
            query_domain=self.query_domain,
            complexity=self.complexity,
            budget_remaining=self.budget,
            accuracy=accuracy,
            last_accuracy=(
                self.accuracy_history[-2]
                if len(self.accuracy_history) > 1 else 0.001
            ),
            step_number=self._state.step_count,
            accuracy_history=self.accuracy_history.copy(),
            available_models=AVAILABLE_MODELS,
            reward=reward,
            done=done,
        )

    def get_score(self) -> float:
        if self.task_id == 1:
            acc = self.accuracy_history[-1] if self.accuracy_history else 0.001
            score = grade_task1(acc, self.total_cost, EPISODE_BUDGET)

        elif self.task_id == 2:
            score = grade_task2(
                self.accuracy_history, self.total_cost, EPISODE_BUDGET
            )

        else:
            score = grade_task3(
            self.resolved,
            self.accuracy_history,
            self.total_cost,
            EPISODE_BUDGET,
            self._state.step_count,
            MAX_STEPS,
        )

        return safe_score(score)

    @property
    def state(self):
        return SimpleNamespace(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            accuracy_history=self.accuracy_history.copy(),
            total_tokens=self.total_tokens,
            total_cost_usd=self.total_cost,
            budget_remaining=self.budget,
            resolved=self.resolved,
            query=self.query,
            query_domain=self.query_domain,
            complexity=self.complexity,
            score=self.get_score(),
        )
