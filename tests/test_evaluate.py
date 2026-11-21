"""Offline tests of all language configurations and the Gemini judge's input contract."""
import asyncio
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_dataset(tmp_path, language):
    cultures = tuple(ev.language_config(language)['culture_names'])
    columns = ev.language_config(language)['label_columns']
    training = [
        {'image_id': 'e1', 'transcriptions': 'Train example one', **{col: 'misogyny' for col in columns.values()}},
        {'image_id': 'e2', 'transcriptions': 'Train example two', **{col: 'not-misogyny' for col in columns.values()}},
    ]
    queries = [
        {'image_id': 'q1', 'transcriptions': 'Query one', **{col: '1' for col in columns.values()}},
        {'image_id': 'q2', 'transcriptions': 'Query two', **{col: '0' for col in columns.values()}},
    ]
    trainfile = tmp_path/'train.csv'
    queryfile = tmp_path/'queries.csv'
    write_csv(trainfile, training)
    write_csv(queryfile, queries)
    retrieved = [
        {'image_id': 'q1', 'retrieved_ids': {c: ['e1'] if c != cultures[-1] else ['e2','e1'] for c in cultures}},
        {'image_id': 'q2', 'retrieved_ids': {c: ['e2'] for c in cultures}},
    ]
    return trainfile, queryfile, retrieved, cultures


@pytest.mark.parametrize('language,expected', [
    ('tamil', ('original','irish','chinese')),
    ('malayalam', ('original','irish','chinese')),
    ('english', ('indian','chinese')),
    ('chinese', ('original','indian','irish')),
])
def test_language_config_and_validation(tmp_path, language, expected):
    trainfile, queryfile, records, cultures = synthetic_dataset(tmp_path, language)
    assert cultures == expected
    clean = ev.validate_submission(records, {'q1','q2'}, {'e1','e2'}, 'a', cultures)
    assert len(clean['q1']['retrieved_ids'][cultures[-1]]) == 2
    assert set(clean['q1']['retrieved_ids']) == set(cultures)
    assert len(cultures) in (2,3)
    assert ev.load_csv_records(trainfile, required=('image_id','transcriptions', *ev.language_config(language)['label_columns'].values()))


@pytest.mark.parametrize('language', list(ev.LANGUAGE_CONFIGS))
def test_task_b_scoring(tmp_path, language):
    trainfile, queryfile, raw, cultures = synthetic_dataset(tmp_path, language)
    gold = ev.load_csv_records(queryfile, required=('image_id', *ev.language_config(language)['label_columns'].values()))
    for row in raw:
        row['predicted_labels'] = {c: ev.label_as_integer(
            gold[row['image_id']][ev.language_config(language)['label_columns'][c]],
            qid=row['image_id'], culture=c) for c in cultures}
    submission = ev.validate_submission(raw, set(gold), {'e1','e2'}, 'b', cultures)
    result = ev.score_task_b(submission, gold, language=language)
    assert result['score'] == 1.0
    assert result['language'] == language
    assert set(result['per_culture']) == set(cultures)
    assert all(v['macro_f1'] == 1.0 for v in result['per_culture'].values())


@pytest.mark.parametrize('language', list(ev.LANGUAGE_CONFIGS))
def test_aggregation_query_balanced(tmp_path, language):
    _, _, raw, cultures = synthetic_dataset(tmp_path, language)
    submission = ev.validate_submission(raw, {'q1','q2'}, {'e1','e2'}, 'a', cultures)
    judgments = {}
    for qid, record in submission.items():
        for c in cultures:
            for evidence in record['retrieved_ids'][c]:
                grade = 0 if qid=='q1' and c==cultures[-1] and evidence=='e2' else 2
                judgments[(qid,c,evidence)] = {key:grade for key in ev.GRADE_KEYS} | {'justification':'Synthetic test'}
    result = ev.aggregate_task_a(submission, judgments, language=language)
    assert result['per_culture'][cultures[-1]]['overall_relevance'] == pytest.approx(.75)
    assert result['score'] == pytest.approx(((len(cultures)-1)*1 + .75)/len(cultures))
    assert result['judged_query_evidence_pairs'] == len(cultures)*2+1
    assert result['language'] == language


@pytest.mark.parametrize('language', list(ev.LANGUAGE_CONFIGS))
def test_wrong_culture_keys_rejected(tmp_path, language):
    _, _, raw, cultures = synthetic_dataset(tmp_path, language)
    raw[0]['retrieved_ids']['wrong'] = ['e1']
    with pytest.raises(ev.EvaluationError):
        ev.validate_submission(raw, {'q1','q2'}, {'e1','e2'}, 'a', cultures)


@pytest.mark.parametrize('language', list(ev.LANGUAGE_CONFIGS))
def test_missing_or_duplicate_evidence_rejected(tmp_path, language):
    _, _, raw, cultures = synthetic_dataset(tmp_path, language)
    raw[0]['retrieved_ids'][cultures[0]] = []
    with pytest.raises(ev.EvaluationError):
        ev.validate_submission(raw, {'q1','q2'}, {'e1','e2'}, 'a', cultures)
    raw[0]['retrieved_ids'][cultures[0]] = ['e1','e1']
    with pytest.raises(ev.EvaluationError):
        ev.validate_submission(raw, {'q1','q2'}, {'e1','e2'}, 'a', cultures)


def test_task_b_rejects_string_predictions(tmp_path):
    _, _, raw, cultures = synthetic_dataset(tmp_path, 'english')
    for record in raw:
        record['predicted_labels']={c:0 for c in cultures}
    raw[0]['predicted_labels']['indian']='0'
    with pytest.raises(ev.EvaluationError):
        ev.validate_submission(raw, {'q1','q2'}, {'e1','e2'}, 'b', cultures)


@pytest.mark.parametrize('language', list(ev.LANGUAGE_CONFIGS))
def test_gemini_judge_contract_and_cache(tmp_path, language):
    trainfile, queryfile, raw, cultures = synthetic_dataset(tmp_path, language)
    clean = ev.validate_submission(raw, {'q1','q2'}, {'e1','e2'}, 'a', cultures)
    train = ev.load_csv_records(trainfile, required=('image_id','transcriptions'))
    queries = ev.load_csv_records(queryfile, required=('image_id','transcriptions'))
    query_imgs = {}; train_imgs = {}
    for key in queries:
        path = tmp_path/(key+'.png');path.write_bytes(b'fake-query-image'+key.encode())
        query_imgs[key] = path
    for key in train:
        path = tmp_path/(key+'.png');path.write_bytes(b'fake-evidence-image'+key.encode())
        train_imgs[key] = path
    calls = []
    class MockMetric:
        async def ascore(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(value=1.0, reason=json.dumps({
                'semantic_relevance':2, 'cultural_relevance':2, 'multimodal_relevance':2,
                'overall_relevance':2, 'justification':'Both memes are relevant'}))
    kwargs = dict(submission=clean, query_rows=queries, train_rows=train,
        query_images=query_imgs, train_images=train_imgs,
        cache_path=tmp_path/'cache.jsonl', output_judgments=tmp_path/'judgments.jsonl',
        metric=MockMetric(), model='mock', thinking='MEDIUM', language=language)
    result = asyncio.run(ev.score_task_a(**kwargs))
    assert result['score']==1.0
    assert len(calls)==len(cultures)*2+1
    assert set(calls[0])=={'query_image', 'evidence_image', 'query_transcription', 'evidence_transcription', 'culture_name'}
    assert {c['culture_name'] for c in calls} == set(ev.language_config(language)['culture_names'].values())
    asyncio.run(ev.score_task_a(**kwargs))
    assert len(calls)==len(cultures)*2+1, 'All repeated judgments should use cache'


def test_ragas_custom_metric_interface_mock(monkeypatch):
    metric_mod = ModuleType('ragas.metrics.collections.base')
    class BaseMetric: pass
    class MetricResult:
        def __init__(self, value, reason): self.value, self.reason = value, reason
    metric_mod.BaseMetric, metric_mod.MetricResult = BaseMetric, MetricResult
    for name in ['ragas', 'ragas.metrics', 'ragas.metrics.collections']:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, 'ragas.metrics.collections.base', metric_mod)
    class FakeGemini:
        def grade(self, **kwargs):
            return {k:2 for k in ev.GRADE_KEYS} | {'justification':'Relevant'}
    metric = ev.build_ragas_metric(FakeGemini())
    value = asyncio.run(metric.ascore(query_transcription='Query',culture_name='Irish (Ireland)'))
    assert value.value==1.0


def test_uploaded_test_schemas():
    root = Path('/mnt/data')
    test_specs = [
        ('tamil', 'test.csv', 356),
        ('english', 'test (1).csv', 1000),
        ('chinese', 'test (2).csv', 340),
    ]
    for lang, filename, count in test_specs:
        if not (root / filename).is_file():
            pytest.skip('Uploaded source CSVs not available in this environment')
        req=('image_id','transcriptions',*ev.language_config(lang)['label_columns'].values())
        rows=ev.load_csv_records(root/filename,required=req)
        assert len(rows)==count
