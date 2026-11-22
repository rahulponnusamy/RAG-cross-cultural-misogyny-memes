# Multilingual meme evaluation

This repository evaluates systems for cross-cultural misogyny detection in Tamil, Malayalam, English, and Chinese memes.

- **Task A** evaluates the relevance of retrieved training memes using a Gemini judge.
- **Task B** evaluates predicted misogyny labels using Macro-F1. It validates the retrieved evidence IDs but does not include them in the Task B score.

## 1. Set up the environment

From the repository root, create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.11 was used for the local setup. Activate the environment each time you open a new terminal.

## 2. Prepare the dataset

Place each language's dataset in this structure:

```text
datasets/<language>/
  train/
    train.csv
    <training images>
  development/
    development.csv
    <development images>
  test/
    test.csv
    <test images>
```

The evaluator discovers these files automatically. The existing Tamil folder layout is also supported. The query and training CSVs require `image_id` and `transcriptions` columns. The training CSV also needs the label columns shown below. Query and training IDs must be distinct.

For Task A, image filenames must match their `image_id`, such as `123.jpg`. JPG, JPEG, PNG, and WEBP images are supported, including images in nested folders.

Build your retrieval system using the training set. Evaluate and improve it using the labeled development set. The evaluator uses development by default and reads its label columns for Task B scoring. Once your system is ready, generate a submission for the test set, which contains query IDs and transcriptions without labels. Task B scoring requires labeled gold data; organizers can supply it using `--gold`.

| Language | Culture keys | Training and gold label columns |
| --- | --- | --- |
| `tamil` | `original`, `irish`, `chinese` | `original_labels`, `irish_labels`, `chinese_labels` |
| `malayalam` | `original`, `irish`, `chinese` | `original_labels`, `irish_labels`, `chinese_labels` |
| `english` | `indian`, `chinese` | `indian_labels`, `chinese_labels` |
| `chinese` | `original`, `indian`, `irish` | `original_labels`, `indian_labels`, `irish_labels` |

For Tamil and Malayalam, `original` refers to the Indian perspective. For Chinese, it refers to the Chinese perspective.

## 3. Prepare a submission

Provide a JSON array or a JSONL file containing one record per query. Each required culture needs at least one retrieved ID from the training CSV. The same list cannot contain duplicate IDs. Evidence can be listed in relevance order, but ranking is not scored.

A Tamil Task A submission has this shape:

```json
[
  {
    "image_id": "QUERY_ID",
    "retrieved_ids": {
      "original": ["TRAIN_1", "TRAIN_2"],
      "irish": ["TRAIN_3"],
      "chinese": ["TRAIN_4"]
    }
  }
]
```

For Task B, add numeric `predicted_labels` for every culture. Use `1` for misogyny and `0` for not misogyny:

```json
[
  {
    "image_id": "QUERY_ID",
    "retrieved_ids": {
      "original": ["TRAIN_1"],
      "irish": ["TRAIN_2"],
      "chinese": ["TRAIN_3"]
    },
    "predicted_labels": {
      "original": 1,
      "irish": 0,
      "chinese": 1
    }
  }
]
```

Task B may also include an `explanations` object with a string for each culture. Explanations are not scored. Replace the placeholder IDs with real dataset IDs. See the [examples folder](examples/) for each language and task.

## 4. Evaluate Task A

Task A requires Gemini access. To use Google Cloud Vertex AI, configure Application Default Credentials and an enabled Vertex AI API:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=true
```

Replace `YOUR_PROJECT_ID` with your Google Cloud project ID. Alternatively, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` for the Gemini API. Keep credentials private.

Run the evaluator with the language and your Task A submission:

```bash
python evaluate.py --language tamil --task a --submission path/to/task_a.json
```

For a partial trial, add `--trial`. The evaluator selects the submitted query IDs from the development CSV in memory:

```bash
python evaluate.py --language tamil --task a --submission path/to/task_a.json --trial
```

Task A defaults to `gemini-3.8-flash`, `LOW` thinking, and one concurrent judgment. The progress bar counts query–evidence pairs. Scores and detailed judgments are saved under `outputs/runs/`, and the reusable cache is saved under `outputs/cache/`. The terminal prints the score file path.

To process more requests at once, add `--concurrency 5`. Available capacity and quota can limit the speedup. Cached judgments are reused on later runs with matching settings and inputs.

## 5. Evaluate Task B

With development labels in the development CSV, run:

```bash
python evaluate.py --language tamil --task b --submission path/to/task_b.json
```

Add `--trial` for a partial submission. Task B does not call Gemini. Its result is saved under `outputs/runs/`.

You can check either submission without scoring or Gemini access:

```bash
python evaluate.py --language tamil --task a --submission path/to/task_a.json --validate-only
```

Add `--trial` to validate a subset. Validation checks the submission and CSVs; it does not check images or API access. To validate a test submission, add `--queries datasets/tamil/test/test.csv`. For Task A scoring against that set, also provide `--query-images datasets/tamil/test/`.

## How scoring works

For every query, culture, and retrieved training meme, Task A sends the query image and transcription, retrieved image and transcription, and target cultural perspective to Gemini. Classification labels are not sent. The judge gives four relevance grades from 0 to 2 and a justification. The primary score normalizes the overall grade to 0–1, averages evidence within each query and culture, then averages queries and cultures equally. Rank is not scored.

Task B calculates Macro-F1 for each culture using the predicted and gold labels, then takes an unweighted mean across cultures. Gold labels may be `0`/`1` or `not-misogyny`/`misogyny`.

## Command arguments

Use `python evaluate.py --help` to see the options in the terminal. Paths passed explicitly are relative to your current working directory.

| Argument | Purpose | Default |
| --- | --- | --- |
| `--language` | Select `tamil`, `malayalam`, `english`, or `chinese`. | Required |
| `--task` | Select `a` for retrieval or `b` for classification. | Required |
| `--submission` | Path to the submission JSON or JSONL. | Required |
| `--trial` | Allow a submission covering a subset of query IDs. | Full query coverage required |
| `--validate-only` | Validate the submission and count pairs without scoring. | Scoring enabled |
| `--queries` | Select another query CSV, such as the test CSV. | `datasets/<language>/development/development.csv` |
| `--train` | Override the training CSV. | `datasets/<language>/train/train.csv` |
| `--gold` | Select a labeled CSV for Task B scoring. | Selected query CSV |
| `--query-images` | Override the Task A query image folder. | `datasets/<language>/development/` |
| `--train-images` | Override the Task A training image folder. | `datasets/<language>/train/` |
| `--model` | Set the Gemini model for Task A. | `gemini-3.8-flash` |
| `--thinking` | Set the thinking level to `LOW`, `MEDIUM`, or `HIGH`. | `LOW` |
| `--concurrency` | Set the maximum number of simultaneous Task A judgments. | `1` |
| `--cache` | Set the reusable Task A judgment cache path. | `outputs/cache/judge_cache.jsonl` |
| `--judgments` | Set the Task A detailed judgments JSONL path. | Timestamped directory under `outputs/runs/` |
| `--output` | Set the score JSON path. | Timestamped directory under `outputs/runs/` |
| `--max-judge-pairs` | Reject Task A submissions exceeding this number of pairs. | `0`, meaning no limit |

Default data paths are resolved relative to the repository. If the standard Tamil development or train CSV is missing, the evaluator uses the existing `datasets/tamil/train & dev/dev/dev.csv` or `datasets/tamil/train & dev/train/train.csv` respectively. For a fresh Task A judgment of the same inputs and settings, supply a new `--cache` path.
