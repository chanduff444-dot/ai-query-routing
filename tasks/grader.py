"""
Grader — three task-level scoring functions.
Returns float strictly in (0.0, 1.0).
"""


def grade_task1(accuracy: float, cost: float, budget: float) -> float:
    """
    Task 1 — Single-Turn Model Routing.
    Score = accuracy if cost within budget else 0.001
    Range: 0.501 – 0.999
    """
    if cost > budget:
        return 0.001
    return round(max(0.501, min(0.999, accuracy)), 4)


def grade_task2(accuracy_history: list, total_cost: float, budget: float) -> float:
    """
    Task 2 — Budget-Constrained Multi-Query Allocation.
    Score = mean_accuracy * budget_compliance_factor
    Range: 0.301 – 0.999
    """
    if not accuracy_history:
        return 0.001
    mean_acc = sum(accuracy_history) / len(accuracy_history)
    if total_cost <= budget:
        compliance = 1.0
    elif total_cost <= budget * 1.2:
        compliance = 0.7
    else:
        compliance = 0.3
    return round(max(0.301, min(0.999, mean_acc * compliance)), 4)


def grade_task3(
    resolved: bool,
    accuracy_history: list,
    total_cost: float,
    budget: float,
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    Task 3 — Multi-Step Tool-Augmented Pipeline.
    Score = resolved_ratio * accuracy_bonus - cost_penalty
    +0.2 bonus if completed under 60% of max budget
    Range: 0.001 – 0.999
    """
    if not accuracy_history:
        return 0.001
    resolved_ratio = 1.0 if resolved else (steps_taken / max_steps)
    mean_acc = sum(accuracy_history) / len(accuracy_history)
    cost_penalty = max(0.0, (total_cost - budget) * 0.5)
    score = resolved_ratio * mean_acc - cost_penalty
    if resolved and total_cost <= budget * 0.6:
        score += 0.2
    return round(max(0.001, min(0.999, score)), 4)