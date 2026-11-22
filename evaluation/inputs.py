"""Read and validate evaluation datasets, submissions, and images."""

import csv
import json
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class EvaluationError(ValueError):
    pass


def _identifier(value: Any, description: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EvaluationError(f"{description} must be a nonempty string or integer ID")
    value = str(value)
    if not value or value != value.strip():
        raise EvaluationError(f"{description}: invalid ID {value!r}")
    return value


def load_csv_records(path: Path, *, required: tuple[str, ...]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise EvaluationError(f"{path}: required CSV columns are {required}")
        rows: dict[str, dict[str, str]] = {}
        for line, row in enumerate(reader, 2):
            rid = _identifier(row.get("image_id"), f"{path}:{line}: image_id")
            if rid in rows:
                raise EvaluationError(f"{path}: duplicate image_id {rid!r}")
            if "transcriptions" in required and (row.get("transcriptions") is None or not str(row["transcriptions"]).strip()):
                raise EvaluationError(f"{path}:{line}: missing transcription for {rid}")
            rows[rid] = row
    if not rows:
        raise EvaluationError(f"{path}: no records")
    return rows


def read_submission(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            data = [json.loads(line) for line in stream if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise EvaluationError("Submission must be a JSON array (or a .jsonl file)")
    return data


def validate_submission(
    submitted: list[Any], query_ids: set[str], train_ids: set[str], task: str,
    cultures: tuple[str, ...], *, require_full: bool = True,
) -> dict[str, dict[str, Any]]:
    """Validate independent full-language Task A/Task B submissions.

    Every culture must have >=1 retrieved ID; list length is participant-chosen.
    No ranking metric is used, but the original ID order is preserved.
    """
    output: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(submitted, 1):
        if not isinstance(record, dict):
            raise EvaluationError(f"Submission record {position} is not an object")
        keys_allowed = {"image_id", "retrieved_ids"} if task == "a" else {"image_id", "retrieved_ids", "predicted_labels", "explanations"}
        if set(record) - keys_allowed:
            raise EvaluationError(f"Record {position}: unexpected fields: {sorted(set(record) - keys_allowed)}")
        qid = _identifier(record.get("image_id"), f"record {position} image_id")
        if qid not in query_ids:
            raise EvaluationError(f"Unknown query image_id: {qid}")
        if qid in output:
            raise EvaluationError(f"Duplicate query image_id: {qid}")
        retrieved = record.get("retrieved_ids")
        if not isinstance(retrieved, dict) or set(retrieved) != set(cultures):
            raise EvaluationError(f"{qid}: retrieved_ids must contain exactly {cultures}")
        clean_retrieval = {}
        for culture in cultures:
            ids = retrieved[culture]
            if not isinstance(ids, list) or len(ids) < 1:
                raise EvaluationError(f"{qid}/{culture}: provide at least one evidence ID")
            valid_ids = [_identifier(eid, f"{qid}/{culture} evidence ID") for eid in ids]
            if len(valid_ids) != len(set(valid_ids)):
                raise EvaluationError(f"{qid}/{culture}: duplicate evidence IDs")
            invalid = set(valid_ids) - train_ids
            if invalid:
                raise EvaluationError(f"{qid}/{culture}: evidence IDs not in the official training corpus: {sorted(invalid)[:5]}")
            clean_retrieval[culture] = valid_ids
        clean = {"image_id": qid, "retrieved_ids": clean_retrieval}
        if task == "b":
            labels = record.get("predicted_labels")
            if not isinstance(labels, dict) or set(labels) != set(cultures):
                raise EvaluationError(f"{qid}: predicted_labels must contain exactly {cultures}")
            for culture, v in labels.items():
                if type(v) is not int or v not in (0, 1):
                    raise EvaluationError(f"{qid}/{culture}: predicted label must be numeric 0 or 1")
            clean["predicted_labels"] = labels.copy()
            if "explanations" in record:
                exps = record["explanations"]
                if not isinstance(exps, dict) or set(exps) != set(cultures) or any(not isinstance(v, str) for v in exps.values()):
                    raise EvaluationError(f"{qid}: explanations (if present) must contain strings for all cultures")
        output[qid] = clean
    missing = query_ids - set(output)
    if require_full and missing:
        raise EvaluationError(f"Submission is missing {len(missing)} query IDs; sample: {sorted(missing)[:5]}")
    return output


def index_images(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise EvaluationError(f"Image directory not found: {folder}")
    found: dict[str, Path] = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in found:
                raise EvaluationError(f"Ambiguous image filenames for image_id {path.stem}: {found[path.stem]}, {path}")
            found[path.stem] = path
    return found


def check_image_ids(image_index: dict[str, Path], needed: set[str], description: str) -> None:
    missing = needed - set(image_index)
    if missing:
        raise EvaluationError(f"Missing {len(missing)} {description} images (filenames must match image_id), e.g. {sorted(missing)[:5]}")
