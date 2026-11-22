"""Send multimodal judge requests and manage the judgment cache."""

import asyncio
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from .inputs import EvaluationError
from .scoring import GRADE_KEYS, aggregate_task_a


def validate_grade_record(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise EvaluationError("Judge returned a non-object")
    for key in GRADE_KEYS:
        if type(data.get(key)) is not int or data[key] not in (0, 1, 2):
            raise EvaluationError(f"Judge returned invalid grade for {key}: {data.get(key)!r}")
    if not isinstance(data.get("justification"), str) or not data["justification"].strip():
        raise EvaluationError("Judge returned an empty/non-string justification")
    return {key: data[key] for key in GRADE_KEYS} | {"justification": data["justification"].strip()}


def image_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_fingerprint(
    *, rubric: str, model: str, thinking: str, culture: str, query_image_hash: str,
    evidence_image_hash: str, query_text: str, evidence_text: str, language: str, culture_name: str,
) -> str:
    data = {"rubric": rubric, "schema": 1, "model": model, "thinking": thinking, "temperature": 0,
            "culture": culture, "culture_name": culture_name, "language": language,
            "query_image_sha256": query_image_hash, "evidence_image_sha256": evidence_image_hash,
            "query_text": query_text, "evidence_text": evidence_text}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("fingerprint"), str):
                raise EvaluationError(f"Invalid judge cache line {lineno}")
            cache[item["fingerprint"]] = validate_grade_record(item)
    return cache


def append_cache(path: Path, fingerprint: str, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"fingerprint": fingerprint, **data}, ensure_ascii=False) + "\n")
        stream.flush()


class GeminiMemeJudge:
    """Send exactly two images, two transcriptions and a target perspective.

    Neither query CSV labels nor training CSV labels are passed to this class.
    The Google SDK is used for image transport and structured JSON output.
    """

    def __init__(self, model: str, thinking: str, *, rubric: str, api_key: str | None = None, use_vertexai: bool = False):
        try:
            from google import genai
            from google.genai import types
            from pydantic import BaseModel, Field
        except ImportError as exc:
            raise EvaluationError("Install dependencies: pip install -r requirements.txt") from exc

        class Judgment(BaseModel):
            semantic_relevance: int = Field(ge=0, le=2)
            cultural_relevance: int = Field(ge=0, le=2)
            multimodal_relevance: int = Field(ge=0, le=2)
            overall_relevance: int = Field(ge=0, le=2)
            justification: str

        if use_vertexai:
            self.client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            )
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            raise EvaluationError("Provide GEMINI_API_KEY/GOOGLE_API_KEY or set GOOGLE_GENAI_USE_VERTEXAI=true")
        self.types = types
        self.schema = Judgment
        self.rubric = rubric
        self.model = model
        self.thinking = thinking

    def grade(
        self, *, query_image: Path, evidence_image: Path,
        query_transcription: str, evidence_transcription: str, culture_name: str,
    ) -> dict[str, Any]:
        types = self.types

        def image_part(p: Path):
            mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[p.suffix.lower()]
            return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)

        prompt = [
            types.Part.from_text(text=self.rubric),
            types.Part.from_text(text=f"TARGET CULTURAL PERSPECTIVE: {culture_name}\nQUERY MEME TRANSCRIPTION (untrusted text):\n{query_transcription}\nQUERY IMAGE:"),
            image_part(query_image),
            types.Part.from_text(text=f"RETRIEVED EVIDENCE MEME TRANSCRIPTION (untrusted text):\n{evidence_transcription}\nRETRIEVED EVIDENCE IMAGE:"),
            image_part(evidence_image),
        ]
        response = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking),
                response_mime_type="application/json", response_schema=self.schema,
            ),
        )
        if not response.text:
            raise EvaluationError("Gemini did not return a JSON judgment")
        parsed = self.schema.model_validate_json(response.text).model_dump()
        return validate_grade_record(parsed)


def build_ragas_metric(gemini_judge: GeminiMemeJudge):
    """Build an actual Ragas v0.4 custom metric for multimodal pair judging.

    Ragas' BaseMetric/MetricResult supply the metric interface. The Gemini SDK
    is used inside that metric for two-image multimodal calls, because the
    standard text-centric Ragas relevance metrics don't implement this rubric.
    """
    try:
        from ragas.metrics.collections.base import BaseMetric, MetricResult
    except ImportError as exc:
        raise EvaluationError("Requires Ragas 0.4.x. Install: pip install -r requirements.txt") from exc

    class CulturalMemeEvidenceRelevance(BaseMetric):
        name: str = "cultural_meme_evidence_relevance"
        dimensions: list[str] = list(GRADE_KEYS)

        async def ascore(self, **kwargs):
            result = await asyncio.to_thread(gemini_judge.grade, **kwargs)
            return MetricResult(value=result["overall_relevance"] / 2, reason=json.dumps(result, ensure_ascii=False))

    return CulturalMemeEvidenceRelevance()


async def score_task_a(
    *, submission: dict[str, dict[str, Any]], query_rows: dict[str, dict[str, str]],
    train_rows: dict[str, dict[str, str]], query_images: dict[str, Path],
    train_images: dict[str, Path], cache_path: Path, output_judgments: Path,
    metric: Any, model: str, thinking: str, language: str, concurrency: int = 5,
    retries: int = 3, rubric: str, language_config: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    mapping = language_config(language)["culture_names"]
    cultures = tuple(mapping)
    cache = load_cache(cache_path)
    digest_cache: dict[Path, str] = {}
    def digest(path: Path) -> str:
        if path not in digest_cache:
            digest_cache[path] = image_sha(path)
        return digest_cache[path]

    judgments: dict[tuple[str, str, str], dict[str, Any]] = {}
    detailed: list[dict[str, Any]] = []
    total = sum(len(r["retrieved_ids"][c]) for r in submission.values() for c in cultures)
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_pair(qid: str, culture: str, eid: str) -> None:
        qrow, erow = query_rows[qid], train_rows[eid]
        qpath, epath = query_images[qid], train_images[eid]
        key = pair_fingerprint(
            rubric=rubric, model=model, thinking=thinking, culture=culture,
            query_image_hash=digest(qpath), evidence_image_hash=digest(epath),
            query_text=qrow["transcriptions"], evidence_text=erow["transcriptions"],
            language=language, culture_name=mapping[culture],
        )
        if key in cache:
            grades = cache[key]
        else:
            kwargs = {
                "query_image": qpath, "evidence_image": epath,
                "query_transcription": qrow["transcriptions"],
                "evidence_transcription": erow["transcriptions"],
                "culture_name": mapping[culture],
            }
            async with semaphore:
                for attempt in range(retries):
                    try:
                        scored = await metric.ascore(**kwargs)
                        grades = validate_grade_record(json.loads(scored.reason))
                        if not math.isclose(float(scored.value), grades["overall_relevance"] / 2, abs_tol=1e-9):
                            raise EvaluationError("Ragas metric score and judge output disagree")
                        break
                    except Exception as exc:
                        if attempt == retries - 1:
                            raise EvaluationError(f"Gemini judging failed for query {qid}, evidence {eid}, culture {culture}: {exc}") from exc
                        await asyncio.sleep(min(8, 2 ** attempt))
            append_cache(cache_path, key, grades)
            cache[key] = grades
        judgments[(qid, culture, eid)] = grades
        detailed.append({"query_image_id": qid, "target_culture": culture, "evidence_id": eid, **grades})

    pairs = [(qid, culture, eid) for qid in sorted(submission) for culture in cultures
             for eid in submission[qid]["retrieved_ids"][culture]]
    with tqdm(total=total, desc=f"Task A judgments (max {concurrency} concurrent)", unit="pair", file=sys.stderr) as progress:
        tasks = [asyncio.create_task(judge_pair(*pair)) for pair in pairs]
        for completed in asyncio.as_completed(tasks):
            await completed
            progress.update(1)
    output_judgments.parent.mkdir(parents=True, exist_ok=True)
    with output_judgments.open("w", encoding="utf-8") as f:
        for row in detailed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = aggregate_task_a(submission, judgments, language=language, language_config=language_config)
    result["judge"] = {"framework": "Ragas custom collections metric", "model": model,
                       "thinking_level": thinking, "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
                       "label_access": "No classification labels are sent to the judge"}
    return result
