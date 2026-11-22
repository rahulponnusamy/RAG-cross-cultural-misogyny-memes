"""Compute retrieval relevance and classification scores."""

from typing import Any, Callable

from .inputs import EvaluationError

GRADE_KEYS = ("semantic_relevance", "cultural_relevance", "multimodal_relevance", "overall_relevance")


def label_as_integer(value: str, *, qid: str, culture: str) -> int:
    mapping = {"1": 1, "0": 0, "misogyny": 1, "not-misogyny": 0}
    normalized = (value or "").strip().lower()
    if normalized not in mapping:
        raise EvaluationError(f"Missing/invalid private gold label at {qid}/{culture}: {value!r}")
    return mapping[normalized]


def score_task_b(
    submission: dict[str, dict[str, Any]], gold: dict[str, dict[str, str]],
    *, language: str, language_config: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    if set(gold) != set(submission):
        raise EvaluationError("Gold query IDs do not match the submitted query IDs")
    config = language_config(language)
    cultures = tuple(config["culture_names"])
    scores = {}
    for culture in cultures:
        column = config["label_columns"][culture]
        if column not in next(iter(gold.values())):
            raise EvaluationError(f"Private gold file lacks {column}")
        ids = sorted(submission)
        y_true = [label_as_integer(gold[qid][column], qid=qid, culture=culture) for qid in ids]
        y_pred = [submission[qid]["predicted_labels"][culture] for qid in ids]
        scores[culture] = {
            "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        }
    return {
        "task": "B", "language": language, "cultures": list(cultures),
        "primary_metric": "mean_macro_f1_across_cultures",
        "score": sum(scores[c]["macro_f1"] for c in cultures) / len(cultures),
        "per_culture": scores,
        "evaluated_queries": len(submission),
        "note": "Retrieved evidence IDs validated but not part of Task B's primary score.",
    }


def aggregate_task_a(
    submission: dict[str, dict[str, Any]],
    judgments: dict[tuple[str, str, str], dict[str, Any]], *, language: str, language_config: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Equal weight per query, then equal weight across cultural perspectives.

    Each query/culture receives the mean of its retrieved evidence grades.
    Number and order of evidence IDs do not change their relative weights.
    """
    cultures = tuple(language_config(language)["culture_names"])
    per_culture: dict[str, dict[str, float]] = {}
    for culture in cultures:
        query_scores = {key: [] for key in GRADE_KEYS}
        for qid, record in submission.items():
            evidence = [judgments[(qid, culture, eid)] for eid in record["retrieved_ids"][culture]]
            for key in GRADE_KEYS:
                query_scores[key].append(sum(row[key] / 2 for row in evidence) / len(evidence))
        per_culture[culture] = {key: sum(query_scores[key]) / len(query_scores[key]) for key in GRADE_KEYS}
    return {
        "task": "A", "language": language, "cultures": list(cultures),
        "primary_metric": "mean_llm_judged_overall_evidence_relevance",
        "score": sum(per_culture[c]["overall_relevance"] for c in cultures) / len(cultures),
        "per_culture": per_culture,
        "evaluated_queries": len(submission),
        "judged_query_evidence_pairs": sum(len(s["retrieved_ids"][c]) for s in submission.values() for c in cultures),
        "score_range": [0, 1],
        "note": "Grades 0,1,2 are normalized to [0,1]. Ranking order is not scored. A one-item list may score 1.0 if that item is strongly relevant.",
    }
