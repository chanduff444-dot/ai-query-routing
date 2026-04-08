"""
Grader — three task-level scoring functions.
Returns float strictly in (0.0, 1.0).
"""


def safe_score(score: float) -> float:
    """Clamp any score into the strict open interval (0, 1)."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.001

    if score <= 0.0:
        return 0.001
    if score >= 1.0:
        return 0.999
    return min(0.999, max(0.001, score))


def grade_task1(accuracy: float, cost: float, budget: float) -> float:
    accuracy = safe_score(accuracy)

    if float(cost) > float(budget):
        return 0.001

    score = max(0.501, min(0.999, accuracy))
    return safe_score(score)


def grade_task2(accuracy_history: list, total_cost: float, budget: float) -> float:
    if not accuracy_history:
        return 0.001

    clean_history = [safe_score(x) for x in accuracy_history]
    mean_acc = sum(clean_history) / len(clean_history)

    if float(total_cost) <= float(budget):
        compliance = 0.99
    elif float(total_cost) <= float(budget) * 1.2:
        compliance = 0.7
    else:
        compliance = 0.3

    score = max(0.301, min(0.999, mean_acc * compliance))
    return safe_score(score)


def grade_task3(
    resolved: bool,
    accuracy_history: list,
    total_cost: float,
    budget: float,
    steps_taken: int,
    max_steps: int,
) -> float:

    if not accuracy_history:
        return 0.001

    clean_history = [safe_score(x) for x in accuracy_history]
    mean_acc = sum(clean_history) / len(clean_history)

    if max_steps <= 0:
        resolved_ratio = 0.001
    else:
        resolved_ratio = 0.999 if resolved else min(0.999, max(0.001, steps_taken / max_steps))

    cost_penalty = max(0.0, (float(total_cost) - float(budget)) * 0.5)

    score = resolved_ratio * mean_acc - cost_penalty

    if resolved and float(total_cost) <= float(budget) * 0.6:
        score += 0.2

    return safe_score(score)
