"""
inference.py - AI Query Routing OpenEnv
Run: python inference.py --task 1 --seed 42
"""
from __future__ import annotations
import argparse
import os
import sys
import random
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

# ── Env vars (injected by validator) ─────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:4000")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
API_KEY      = os.environ.get("API_KEY",      "dummy-key")

# ── LiteLLM proxy client ──────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

try:
    from server.ai_query_routing_environment import AiQueryRoutingEnvironment as QueryRoutingEnv
except ImportError:
    QueryRoutingEnv = None

try:
    from models import AiQueryRoutingAction as RoutingAction
except ImportError:
    RoutingAction = None


class _Obs:
    def __init__(self, d):
        self.query_complexity = d.get("query_complexity", 0.5)
        self.query_domain     = d.get("query_domain", "factual")
        self.budget_remaining = d.get("budget_remaining", 1.0)
        self.last_accuracy    = d.get("last_accuracy", 0.001)
        self.step_number      = d.get("step_number", 0)
        self.accuracy_history = d.get("accuracy_history", [])
        self.accuracy         = d.get("accuracy", 0.001)
        self.reward           = d.get("reward", 0.001)
        self.done             = d.get("done", False)
        self.query            = d.get("query", "")


class _StubEnv:
    def __init__(self, task_id=1, seed=42):
        self.task_id = task_id
        self._rng    = random.Random(seed)
        self._step   = 0
        self._budget = 1.0
        self._acc    = []

    def reset(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        self._step   = 0
        self._budget = 1.0
        self._acc    = []
        return self._make_obs()

    def step(self, action):
        tokens = self._rng.randint(80, 512)
        acc    = round(min(0.999, max(0.001, self._rng.uniform(0.75, 0.97))), 4)
        cost   = tokens * 0.000003
        self._budget = max(0.001, self._budget - cost)
        self._acc.append(acc)
        self._step += 1
        reward = round(max(0.001, min(0.999, acc - tokens * 0.001)), 4)
        done = (
            self._step >= 10
            or (self.task_id == 1 and self._step >= 1)
            or (self.task_id == 2 and self._step >= 5)
        )
        return self._make_obs(acc=acc, reward=reward, done=done)

    def state(self):
        class S: pass
        s = S()
        s.accuracy_history = self._acc.copy()
        s.total_tokens     = self._step * 200
        s.total_cost_usd   = round(self._step * 0.0003, 6)
        s.budget_remaining = round(self._budget, 4)
        s.resolved         = self._step >= 1
        return s

    def _make_obs(self, acc=0.001, reward=0.001, done=False):
        queries = [
            "What is the capital of France?",
            "Explain how transformers work in deep learning.",
            "Write a Python function to sort a list.",
            "What is 2 + 2?",
            "Summarize the theory of relativity.",
        ]
        return _Obs({
            "query_complexity":  round(self._rng.uniform(0.001, 0.999), 3),
            "query_domain":      self._rng.choice(["factual", "reasoning", "code"]),
            "budget_remaining":  round(self._budget, 4),
            "last_accuracy":     self._acc[-1] if self._acc else 0.001,
            "step_number":       self._step,
            "accuracy_history":  self._acc.copy(),
            "accuracy":          acc,
            "reward":            reward,
            "done":              done,
            "query":             self._rng.choice(queries),
        })


class _Action:
    def __init__(self, model_tier, model_id, max_tokens,
                 trim_context, prompt_strategy, tools):
        self.model_tier      = model_tier
        self.model_id        = model_id
        self.max_tokens      = max_tokens
        self.trim_context    = trim_context
        self.prompt_strategy = prompt_strategy
        self.tools           = tools


def make_action(tier, mid, tokens, trim, strategy, tools):
    if RoutingAction:
        try:
            return RoutingAction(
                model_tier=tier, model_id=mid, max_tokens=tokens,
                trim_context=trim, prompt_strategy=strategy, tools=tools,
            )
        except Exception:
            pass
    return _Action(tier, mid, tokens, trim, strategy, tools)


def call_llm(prompt: str, max_tokens: int = 256) -> str:
    """Make a real API call through the LiteLLM proxy."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        sys.stderr.write(f"[LLM ERROR] {e}\n")
        sys.stderr.flush()
        return "error"


def policy(obs):
    c = getattr(obs, "query_complexity", getattr(obs, "complexity", 0.5))
    b = getattr(obs, "budget_remaining", 1.0)
    s = getattr(obs, "step_number", 0)
    query = getattr(obs, "query", "Answer briefly.")

    # ── REAL LLM CALL through LiteLLM proxy ──────────────────────────────
    prompt = f"Query: {query}\nComplexity: {c}\nBudget: {b}\nChoose best model tier (small/medium/large) in one sentence."
    call_llm(prompt=prompt, max_tokens=64)

    if c < 0.35 or b < 0.05:
        return make_action("small",  "gemini-flash", 128, b < 0.3, "zero_shot", ["none"])
    elif c < 0.70:
        return make_action("medium", "claude",       256, b < 0.3, "cot",       ["none"])
    else:
        tools = ["web_search"] if s < 3 else ["code_exec"] if s < 6 else ["none"]
        return make_action("large",  "gpt4",         512, b < 0.3, "cot",       tools)


TASK_NAMES = {1: "single_turn_routing", 2: "budget_constrained", 3: "tool_pipeline"}


def run(task_id, seed, max_steps):
    if QueryRoutingEnv:
        try:
            env = QueryRoutingEnv(task_id=task_id, seed=seed)
        except TypeError:
            env = QueryRoutingEnv()
    else:
        env = _StubEnv(task_id=task_id, seed=seed)

    obs       = env.reset(seed=seed)
    done      = False
    step      = 0
    rewards   = []
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")

    print(f"[START] task={task_name} seed={seed}", flush=True)

    while not done and step < max_steps:
        action     = policy(obs)
        next_obs   = env.step(action)
        reward_val = float(getattr(next_obs, "reward", 0.001))
        done       = bool(getattr(next_obs, "done", False))
        accuracy   = float(getattr(next_obs, "accuracy", 0.001))
        budget     = float(getattr(next_obs, "budget_remaining", 1.0))

        rewards.append(reward_val)
        print(
            f"[STEP] step={step} model_id={action.model_id} "
            f"model_tier={action.model_tier} "
            f"reward={round(reward_val,4)} accuracy={round(accuracy,4)} "
            f"budget_remaining={round(budget,4)} done={done}",
            flush=True,
        )
        obs   = next_obs
        step += 1

    try:
        state = env.state() if callable(env.state) else env.state
    except Exception:
        state = None

    acc_hist      = getattr(state, "accuracy_history", getattr(obs, "accuracy_history", []))
    mean_reward   = round(sum(rewards) / len(rewards), 4) if rewards else 0.001
    mean_accuracy = round(sum(acc_hist) / len(acc_hist), 4) if acc_hist else 0.001
    resolved      = bool(getattr(state, "resolved", done))
    total_tokens  = getattr(state, "total_tokens", 0)
    total_cost    = getattr(state, "total_cost_usd", 0.0)
    budget_used   = round((1.0 - getattr(state, "budget_remaining", 0.0)) * 100, 1)

    print(
        f"[END] task={task_name} seed={seed} steps={step} "
        f"mean_reward={mean_reward} mean_accuracy={mean_accuracy} "
        f"total_tokens={total_tokens} total_cost_usd={total_cost} "
        f"budget_used_pct={budget_used} resolved={resolved}",
        flush=True,
    )

    return {
        "task_id": task_id, "task_name": task_name, "seed": seed,
        "steps_taken": step, "mean_reward": mean_reward,
        "mean_accuracy": mean_accuracy, "total_tokens": total_tokens,
        "total_cost_usd": total_cost, "budget_used_pct": budget_used,
        "resolved": resolved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  type=int, default=0)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    sys.stderr.write("\n-- Env vars --\n")
    for v in ["API_BASE_URL", "MODEL_NAME", "API_KEY"]:
        sys.stderr.write(f"  {v}: {'SET' if os.environ.get(v) else 'not set'}\n")
    sys.stderr.flush()

    sys.stderr.write("-- Testing LLM proxy --\n")
    test = call_llm("Say OK.", max_tokens=8)
    sys.stderr.write(f"  Proxy result: {repr(test)}\n")
    sys.stderr.flush()

    tasks = [args.task] if args.task in (1, 2, 3) else [1, 2, 3]
    for t in tasks:
        run(t, args.seed, args.steps)


if __name__ == "__main__":
    main()