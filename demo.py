import json
from pathlib import Path
from tempfile import TemporaryDirectory

from archive.collect import capture
from archive.storage import SqliteArchive
from plans.registry import PlanRegistry
from recordings.batches import read_recording
from recordings.source import RecordingSource
from review.export import export
from review.query import query
from review.retest import plan_retest, verify_retest
from review.verification import verify_export


def require_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f'{label} 预期 {expected!r}，实际 {actual!r}')


def main():
    samples = Path(__file__).resolve().parent / 'samples'
    plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2})
    masks = (plan.catalog.get_mask('x'), plan.catalog.get_mask('y'))
    filenames = ('normal.jsonl', 'survey.jsonl', 'recheck.jsonl')
    expected = ([(0, 1)], [(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1)])

    with TemporaryDirectory() as directory:
        database = Path(directory) / 'shift.sqlite3'
        with SqliteArchive.open(database) as archive:
            ids = []
            reports = []
            actual = []
            accepted_ids = []
            for filename, expected_values in zip(filenames, expected, strict=True):
                with RecordingSource(samples / filename) as source:
                    capture_id = capture(archive, plan, read_recording(source))
                values = query(archive, capture_id, *masks)
                require_equal(values, expected_values, filename)
                ids.append(capture_id)
                actual.append(values)
                accepted_ids.append([frame.frame_id for frame in archive.get(capture_id).evidence.frames])
                reports.append(export(archive, capture_id, *masks))

        with SqliteArchive.open(database) as reopened:
            offline_records = []
            for capture_id, expected_values, original in zip(ids, expected, reports, strict=True):
                current = export(reopened, capture_id, *masks)
                check = verify_export(reopened, original, capture_id, *masks)
                require_equal(query(reopened, capture_id, *masks), expected_values, '离线查询')
                require_equal(current, original, '离线导出')
                require_equal(check.matches, True, '离线复核')
                offline_records.append(current.as_dict())

        scenario = {
            'case_id': 'v8',
            'input': {'recordings': filenames, 'masks': masks, 'accepted_frame_ids': accepted_ids},
            'expected': expected,
            'actual': actual,
            'downstream_expected': expected,
            'downstream_actual': [record['combinations'] for record in offline_records],
            'code_paths': [
                'recordings/batches.py', 'admission/frames.py', 'archive/storage.py',
                'archive/feasibility.py', 'review/projection.py', 'review/export.py', 'review/verification.py',
            ],
            'oracle': '手算二位状态 00、01、10、11，逐物理帧计失配，最多保留一次。',
        }
        print('POINTWISE_SCENARIO ' + json.dumps(scenario, ensure_ascii=False))
        for record in offline_records:
            print(json.dumps(record, ensure_ascii=False))


def retest_main():
    """按固定成果演示预算差异、换班重开与原记录复核。"""
    samples = Path(__file__).resolve().parent / 'samples'
    first_shift = PlanRegistry()
    plan = first_shift.register('station-12', 2, {'x': 1, 'y': 2}, actions={'A': (2, 2), 'B': (3, 1)})

    with TemporaryDirectory() as directory:
        database = Path(directory) / 'retest.sqlite3'
        with SqliteArchive.open(database) as archive:
            with RecordingSource(samples / 'retest.jsonl') as source:
                first_id = capture(archive, plan, read_recording(source))
            tight = plan_retest(archive, first_id, 1, 2, 2)
            guaranteed = plan_retest(archive, first_id, 1, 2, 3)
            require_equal(tight['status'], 'no_guarantee', '预算 2')
            require_equal((guaranteed['status'], guaranteed['worst_cost']), ('ok', 3), '预算 3')
            require_equal(guaranteed['tree']['action_id'], 'B', '根测法')

            second_shift = PlanRegistry()
            relieved = second_shift.register('station-12', 2, {'x': 1, 'y': 2}, actions={'B': (1, 1)})
            with RecordingSource(samples / 'retest.jsonl') as source:
                second_id = capture(archive, relieved, read_recording(source))
            shifted = plan_retest(archive, second_id, 1, 2, 3)
            require_equal(shifted['status'], 'no_guarantee', '换班成果')

            adaptive_plan = PlanRegistry().register(
                'station-12', 3, {'x': 1, 'y': 2},
                actions={'A': (4, 2), 'B': (6, 1), 'C': (7, 1)},
            )
            with RecordingSource(samples / 'retest_adaptive.jsonl') as source:
                adaptive_id = capture(archive, adaptive_plan, read_recording(source))
            adaptive_tight = plan_retest(archive, adaptive_id, 1, 2, 3)
            adaptive = plan_retest(archive, adaptive_id, 1, 2, 4)
            require_equal(adaptive_tight['status'], 'no_guarantee', '自适应预算 3')
            require_equal((adaptive['status'], adaptive['worst_cost']), ('ok', 4), '自适应预算 4')
            require_equal(adaptive['tree']['action_id'], 'B', '自适应根测法')

        with SqliteArchive.open(database) as reopened:
            require_equal(plan_retest(reopened, first_id, 1, 2, 3), guaranteed, '重开重算')
            tampered = json.loads(json.dumps(guaranteed))
            tampered['worst_cost'] = 2
            checks = {
                'original_record': verify_retest(reopened, guaranteed, first_id, 1, 2, 3),
                'json_roundtrip': verify_retest(reopened, json.loads(json.dumps(guaranteed)), first_id, 1, 2, 3),
                'swapped_capture': verify_retest(reopened, guaranteed, second_id, 1, 2, 3),
                'tampered_cost': verify_retest(reopened, tampered, first_id, 1, 2, 3),
            }
            require_equal(
                checks,
                {'original_record': True, 'json_roundtrip': True, 'swapped_capture': False, 'tampered_cost': False},
                '交接复核',
            )

        print('POINTWISE_RETEST ' + json.dumps({
            'case_id': 'v8-retest',
            'input': {
                'recordings': ['retest.jsonl', 'retest_adaptive.jsonl'],
                'masks': [1, 2],
                'actions': [{'A': [2, 2], 'B': [3, 1]}, {'B': [1, 1]}, {'A': [4, 2], 'B': [6, 1], 'C': [7, 1]}],
                'budgets': [2, 3, 4],
            },
            'expected': {
                'budget_2': 'no_guarantee', 'budget_3_worst_cost': 3,
                'shifted': 'no_guarantee', 'adaptive_budget_4_worst_cost': 4,
            },
            'actual': {
                'budget_2': tight['status'], 'budget_3_worst_cost': guaranteed['worst_cost'],
                'shifted': shifted['status'], 'adaptive_budget_4_worst_cost': adaptive['worst_cost'],
            },
            'handover': checks,
            'code_paths': ['plans/actions.py', 'archive/codec.py', 'archive/feasibility.py', 'review/retest.py'],
            'oracle': '原批与路径共享一次失配额度，逐节点最小化最坏累计成本，同价按测法 ID 字典序。',
        }, ensure_ascii=False))
        for record in (tight, guaranteed, shifted, adaptive_tight, adaptive):
            print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
    retest_main()
