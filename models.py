from typing import Literal, List
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AiQueryRoutingAction(Action):
    model_id: str = Field(
        default="gemini-flash",
        description="Model to use: gpt4 / claude / gemini-flash / llama"
    )
    model_tier: Literal["small", "medium", "large"] = Field(
        default="small",
        description="Model size category"
    )
    prompt_strategy: Literal["zero_shot", "cot", "few_shot"] = Field(
        default="zero_shot",
        description="Prompting technique"
    )
    tools: List[str] = Field(
        default_factory=list,
        description="Optional tools: web_search, code_exec"
    )
    trim_context: bool = Field(
        default=False,
        description="Whether to reduce input context"
    )
    max_tokens: int = Field(
        default=512,
        description="Maximum tokens allowed for response"
    )


class AiQueryRoutingObservation(Observation):
    query: str = Field(default="")
    query_domain: str = Field(
        default="factual",
        description="Type of query: factual / reasoning / code"
    )
    complexity: float = Field(
        default=0.5,
        description="Difficulty of the query (0 to 1)"
    )
    budget_remaining: float = Field(
        default=1.0,
        description="Remaining budget"
    )
    accuracy: float = Field(
        default=0.001,
        description="Current step accuracy"
    )
    last_accuracy: float = Field(
        default=0.001,
        description="Previous step accuracy"
    )
    step_number: int = Field(
        default=0,
        description="Current step number"
    )
    accuracy_history: List[float] = Field(
        default_factory=list,
        description="List of past accuracies"
    )
    available_models: List[dict] = Field(
        default_factory=list,
        description="List of available models with cost and quality"
    )
    reward: float = Field(default=0.001)
    done: bool = Field(default=False)