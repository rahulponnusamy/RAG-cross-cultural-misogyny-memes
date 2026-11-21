# LT-EDI @ NAACL 2027, multilingual evaluation scripts

## Four language configurations

Select the language with the mandatory `--language` option. **Do not rename cultural keys or CSV columns.** `original` denotes the *native/source* cultural perspective of that language's dataset, not a universal synonym for `indian`.

| `--language` | Participant JSON cultural keys (`retrieved_ids`, and `predicted_labels` for Task B) | Required private Task B gold columns | Cultural perspective sent to Gemini |
|---|---|---|---|
| `tamil` | `original`, `irish`, `chinese` | `original_labels`, `irish_labels`, `chinese_labels` | Original = Indian; Irish; Chinese |
| `malayalam` | `original`, `irish`, `chinese` | `original_labels`, `irish_labels`, `chinese_labels` | Original = Indian; Irish; Chinese |
| `english` | `indian`, `chinese` | `indian_labels`, `chinese_labels` | Indian; Chinese |
| `chinese` | `original`, `indian`, `irish` | `original_labels`, `indian_labels`, `irish_labels` | Original = Chinese; Indian; Irish |

These settings match the user-provided Tamil, English, and Chinese test CSV headers. Malayalam is configured to match Tamil, as specified; a Malayalam dataset was not supplied for direct inspection.

## Task overview

| Task | Required participant output | Primary metric |
|---|---|---|
| A, Cultural Evidence Retrieval | `image_id` and a `retrieved_ids` dictionary for **every** required culture | Mean LLM-judged overall evidence relevance (0–1) using **Ragas + Gemini** |
| B, Retrieval-Augmented Misogyny Detection | Same `retrieved_ids` plus a `predicted_labels` dictionary for **every** required culture | Unweighted mean of **Macro-F1** scores over the language's target cultures |

Retrieval is restricted to IDs in the **training corpus for the selected language**, with at least one evidence ID for **each** query and culture. There is no fixed upper bound; duplicates *within a cultural list* are invalid. Evidence IDs are submitted as a JSON array in model-estimated relevance order, but **ranking order is not scored**. Participants may enter either task independently.

Both tasks require a full submission for every query in the evaluation CSV. For Task B, predicted labels must be **numeric 0 or 1**, not the strings `"0"`, `"1"`, `"misogyny"`, or `"not-misogyny"`:

- `1`: misogyny
- `0`: not misogyny

The private gold file may use numeric `0`/`1` or string annotations `misogyny`/`not-misogyny`.

## Participant submission examples

See `examples/` for Task A and Task B JSON shapes for all four languages. The ID values shown there are **illustrative placeholders**, not complete valid submissions; replace them with the real query IDs and training IDs for the selected language.

**Tamil and Malayalam Task A**

```json
[{"image_id":"QUERY_ID","retrieved_ids":{"original":["TRAIN_001","TRAIN_002"],"irish":["TRAIN_003"],"chinese":["TRAIN_004"]}}]
```

**English Task B**

```json
[{"image_id":"QUERY_ID","retrieved_ids":{"indian":["TRAIN_001"],"chinese":["TRAIN_002","TRAIN_003"]},"predicted_labels":{"indian":1,"chinese":0}}]
```

**Chinese Task A**

```json
[{"image_id":"QUERY_ID","retrieved_ids":{"original":["TRAIN_001"],"indian":["TRAIN_002"],"irish":["TRAIN_003"]}}]
```

Task B also permits an optional `explanations` object with exactly the language's culture keys and string explanations. Explanations are **not** currently scored.

## Installation

```bash
pip install -r requirements.txt
```

Task A requires the organizer's Google API key:

```bash
export GEMINI_API_KEY='YOUR_ORGANIZER_KEY'
```

**Never distribute this key to participants or include it in the public evaluation package.** The configured model is `gemini-3.8-flash`, thinking level `MEDIUM`, as previously agreed. Verify that this exact model identifier and thinking configuration are available in your Google account before official deployment. The actual Gemini/Ragas runtime was not tested here.

## CLI examples

Replace dataset paths with the matching language's files. Separate folders for query and training images are required for Task A. Image files may be JPG/JPEG/PNG/WEBP, named by `image_id`, and may be in nested folders.

**Validate Task A submissions without API usage (English):**

```bash
python evaluate.py --language english --task a --submission english_task_a.json \
  --queries english/test.csv --train english/train.csv --validate-only
```

**Evaluate Task A (Tamil):**

```bash
python evaluate.py --language tamil --task a --submission tamil_task_a.json \
  --queries tamil/test.csv --train tamil/train.csv \
  --query-images tamil/test_images --train-images tamil/train_images \
  --model gemini-3.8-flash --thinking MEDIUM \
  --cache private/tamil_judge_cache.jsonl \
  --judgments private/tamil_judgments.jsonl \
  --output tamil_task_a_scores.json
```

**Evaluate Task A (Malayalam):**

```bash
python evaluate.py --language malayalam --task a --submission malayalam_task_a.json \
  --queries malayalam/test.csv --train malayalam/train.csv \
  --query-images malayalam/test_images --train-images malayalam/train_images \
  --output malayalam_task_a_scores.json
```

**Evaluate Task B (English):**

```bash
python evaluate.py --language english --task b --submission english_task_b.json \
  --queries english/test_public.csv --train english/train.csv \
  --gold private/english_test_gold.csv --output english_task_b_scores.json
```

**Evaluate Task B (Chinese):**

```bash
python evaluate.py --language chinese --task b --submission chinese_task_b.json \
  --queries chinese/test_public.csv --train chinese/train.csv \
  --gold private/chinese_test_gold.csv --output chinese_task_b_scores.json
```

Only `--language` changes the cultural schema. Both tasks work for all four language choices.

## Multimodal LLM-as-a-Judge methodology

For each submitted (query meme, culture, evidence meme) tuple, the custom Ragas metric sends Gemini **only**:

1. Query test/dev meme image and transcription.
2. Retrieved **training** meme image and transcription.
3. The target culture's human-readable perspective, derived from the `--language` mapping.

**No training or test classification labels are sent to the Gemini judge.** Images are resolved by the `image_id` stem, and only image files and transcriptions enter the judge prompt. The rubric evaluates semantic, cultural, multimodal, and holistic overall evidence relevance on an integer `0,1,2` scale. A concise justification is also recorded for each pair.

The primary Task A result averages the holistic grade divided by two, **first across evidence within each query and culture, then across queries within a culture, and finally equally across the language's cultures (two for English, three for others)**. It ignores rank position, judges evidence relevance rather than classification correctness, and can favor concise submissions with a single very relevant item. The other three grades are diagnostic outputs.

Gemini judgments are cached using a hash that includes **language, cultural perspective, query/evidence images, transcriptions, rubric, model, and thinking configuration**. `--max-judge-pairs N` is an optional organizer processing safety limit: submissions that exceed it **fail validation, rather than being truncated**. A model-based judge may still vary; review its cultural relevance against human annotations before using the results for an official leaderboard.

## Task B methodology

Task B accepts the same evidence ID lists and numerical `predicted_labels`. It reads the language-specific private label columns and calculates class-balanced Macro-F1 for each target culture. The overall score is an **unweighted mean across the required cultures**. Task B does not call Gemini. Evidence IDs are validated but **are not included in the primary Macro-F1 score**.

The public `test.csv` exposed to participants must contain **only** `image_id` and `transcriptions`; keep the gold label columns in an organizer-only CSV. Images are distributed in separate folders. The supplied uploaded test CSVs contain private labels and must not be published unchanged.

## Implementation and deployment status

This is an evaluation reference package, **not yet a Codabench ingestion/scoring wrapper**. Offline unit tests cover four language configurations, full submission validation, private gold Macro-F1, cultural prompt mapping, judgment caching, and mocked multimodal grading. The actual Gemini API, Ragas SDK integration, and image-folder setup must be integration-tested before deployment. Keep organizer API keys and private test labels outside participant downloads.
