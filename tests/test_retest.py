import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from itertools import product
from pathlib import Path

import retest_materials
from archive.artifacts import CaptureNotFound
from archive.collect import CaptureRejected, capture
from archive.storage import ArchiveError, SqliteArchive
from plans.model import PlanConflict
from plans.registry import PlanRegistry
from recordings.batches import read_recording
from recordings.source import RecordingSource
from review.parameters import QueryParameterError
from review.query import query
from review.retest import plan_retest, verify_retest
from tests.test_capture import capture_file, write_recording

SAMPLES = Path(__file__).resolve().parent.parent / 'samples'

RETEST_TREE_B3 = {
    'action_id': 'B',
    'branches': {
        '0': {
            'action_id': 'B',
            'branches': {
                '0': {'value': [0, 0]},
                '1': {
                    'action_id': 'B',
                    'branches': {'0': {'value': [0, 0]}, '1': {'value': [0, 1]}},
                },
            },
        },
        '1': {
            'action_id': 'B',
            'branches': {
                '0': {
                    'action_id': 'B',
                    'branches': {'0': {'value': [0, 0]}, '1': {'value': [0, 1]}},
                },
                '1': {'value': [0, 1]},
            },
        },
    },
}

ADAPTIVE_TREE_B4 = {
    'action_id': 'B',
    'branches': {
        '0': {
            'action_id': 'C',
            'branches': {
                '0': {'value': [0, 0]},
                '1': {
                    'action_id': 'C',
                    'branches': {'0': {'value': [0, 0]}, '1': {'value': [1, 0]}},
                },
            },
        },
        '1': {
            'action_id': 'A',
            'branches': {
                '0': {
                    'action_id': 'B',
                    'branches': {'0': {'value': [0, 0]}, '1': {'value': [0, 1]}},
                },
                '1': {'value': [0, 0]},
            },
        },
    },
}


def state_parity(state, mask):
    return (state & mask).bit_count() % 2


def oracle_plan(width, frames, masks, actions, budget):
    """独立枚举源状态的暴力补采规划，与正式实现对照。"""
    base = {
        state: sum(state_parity(state, mask) != value for mask, value in frames)
        for state in range(1 << width)
    }

    def solve(counts, remaining):
        possible = [state for state, misses in counts.items() if misses <= 1]
        joints = {(state_parity(s, masks[0]), state_parity(s, masks[1])) for s in possible}
        if not joints:
            return None
        if len(joints) == 1:
            only = next(iter(joints))
            return (0, {'value': [only[0], only[1]]})
        best = None
        for action_id in sorted(actions):
            mask, cost = actions[action_id]
            if cost > remaining:
                continue
            branches = {}
            worst = 0
            covered = True
            for reading in (0, 1):
                child = {
                    state: misses + (state_parity(state, mask) != reading)
                    for state, misses in counts.items()
                }
                child = {state: misses for state, misses in child.items() if misses <= 1}
                if not child:
                    continue
                solved = solve(child, remaining - cost)
                if solved is None:
                    covered = False
                    break
                branches[str(reading)] = solved[1]
                worst = max(worst, solved[0])
            if not covered or not branches:
                continue
            total = cost + worst
            if best is None or total < best[0]:
                best = (total, {'action_id': action_id, 'branches': branches})
        return best

    return solve(base, budget)


class ActionRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.registry = PlanRegistry()

    def test_register_actions_sorted_and_copied(self):
        source = {'B': (3, 1), 'A': [2, 2]}
        plan = self.registry.register('station-12', 2, {'x': 1, 'y': 2}, actions=source)
        source['C'] = (1, 1)
        self.assertEqual(plan.actions.items(), (('A', (2, 2)), ('B', (3, 1))))
        again = self.registry.register('station-12', 2, {'y': 2, 'x': 1}, actions={'B': [3, 1], 'A': (2, 2)})
        self.assertIs(again, plan)
        self.assertIsNone(self.registry.register('other', 2, {}).actions)
        self.assertEqual(self.registry.register('empty', 2, {}, actions={}).actions.items(), ())

    def test_register_action_conflict(self):
        plan = self.registry.register('station-12', 2, {}, actions={'A': (1, 1)})
        for actions in ({'A': (1, 2)}, {'B': (1, 1)}, {}, None):
            with self.subTest(actions=actions):
                if actions is None:
                    with self.assertRaises(PlanConflict):
                        self.registry.register('station-12', 2, {})
                else:
                    with self.assertRaises(PlanConflict):
                        self.registry.register('station-12', 2, {}, actions=actions)
        self.assertIs(self.registry.get('station-12'), plan)
        independent = PlanRegistry().register('station-12', 2, {}, actions={'B': (1, 1)})
        self.assertNotEqual(independent, plan)

    def test_register_invalid_actions(self):
        invalid = (
            1, 'A', [('A', (1, 1))], {1: (1, 1)}, {'': (1, 1)}, {None: (1, 1)},
            {'A': 1}, {'A': None}, {'A': (1,)}, {'A': (1, 1, 1)}, {'A': '12'},
            {'A': (0, 1)}, {'A': (4, 1)}, {'A': (-1, 1)}, {'A': (1.0, 1)}, {'A': (True, 1)},
            {'A': (1, 0)}, {'A': (1, 7)}, {'A': (1, -1)}, {'A': (1, 1.0)}, {'A': (1, True)},
            {'A': (1, 1), 'B': (1, 1), 'C': (1, 1), 'D': (1, 1), 'E': (1, 1)},
        )
        for actions in invalid:
            with self.subTest(actions=actions):
                with self.assertRaises(PlanConflict):
                    self.registry.register('unassigned', 2, {}, actions=actions)
                with self.assertRaises(KeyError):
                    self.registry.get('unassigned')


class RetestSampleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.database = self.root / 'retest.sqlite3'

    def capture_sample(self, archive, plan, filename):
        with RecordingSource(SAMPLES / filename) as source:
            return capture(archive, plan, read_recording(source))

    def test_retest_sample_budgets(self):
        plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2}, actions={'A': (2, 2), 'B': (3, 1)})
        with SqliteArchive.open(self.database) as archive:
            capture_id = self.capture_sample(archive, plan, 'retest.jsonl')
            tight = plan_retest(archive, capture_id, 1, 2, 2)
            self.assertEqual(tight, {
                'capture_id': capture_id, 'masks': [1, 2], 'budget': 2,
                'status': 'no_guarantee', 'worst_cost': None, 'tree': None,
            })
            record = plan_retest(archive, capture_id, 1, 2, 3)
            self.assertEqual(record['status'], 'ok')
            self.assertEqual(record['worst_cost'], 3)
            self.assertEqual(record['tree'], RETEST_TREE_B3)
            self.assertTrue(verify_retest(archive, record, capture_id, 1, 2, 3))
            self.assertTrue(verify_retest(archive, json.loads(json.dumps(record)), capture_id, 1, 2, 3))

    def test_retest_sample_tie_prefers_lexicographic(self):
        plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2}, actions={'A': (2, 1), 'B': (3, 1)})
        with SqliteArchive.open(self.database) as archive:
            capture_id = self.capture_sample(archive, plan, 'retest.jsonl')
            record = plan_retest(archive, capture_id, 1, 2, 3)
            self.assertEqual(record['worst_cost'], 3)

            def action_ids(node):
                if 'action_id' not in node:
                    return []
                ids = [node['action_id']]
                for child in node['branches'].values():
                    ids.extend(action_ids(child))
                return ids

            self.assertEqual(action_ids(record['tree']), ['A'] * 5)

    def test_shift_change_and_reopen(self):
        first_shift = PlanRegistry()
        plan = first_shift.register('station-12', 2, {'x': 1, 'y': 2}, actions={'A': (2, 2), 'B': (3, 1)})
        with SqliteArchive.open(self.database) as archive:
            first_id = self.capture_sample(archive, plan, 'retest.jsonl')
            record = plan_retest(archive, first_id, 1, 2, 3)
            second_shift = PlanRegistry()
            relieved = second_shift.register('station-12', 2, {'x': 1, 'y': 2}, actions={'B': (1, 1)})
            second_id = self.capture_sample(archive, relieved, 'retest.jsonl')
            self.assertNotEqual(first_id, second_id)
            self.assertEqual(plan_retest(archive, second_id, 1, 2, 3)['status'], 'no_guarantee')
        with SqliteArchive.open(self.database) as reopened:
            self.assertEqual(plan_retest(reopened, first_id, 1, 2, 3), record)
            self.assertTrue(verify_retest(reopened, record, first_id, 1, 2, 3))
            self.assertFalse(verify_retest(reopened, record, second_id, 1, 2, 3))
            tampered_cost = json.loads(json.dumps(record))
            tampered_cost['worst_cost'] = 2
            self.assertFalse(verify_retest(reopened, tampered_cost, first_id, 1, 2, 3))
            tampered_action = json.loads(json.dumps(record))
            tampered_action['tree']['action_id'] = 'A'
            self.assertFalse(verify_retest(reopened, tampered_action, first_id, 1, 2, 3))

    def test_adaptive_sample(self):
        plan = PlanRegistry().register(
            'station-12', 3, {'x': 1, 'y': 2},
            actions={'A': (4, 2), 'B': (6, 1), 'C': (7, 1)},
        )
        with SqliteArchive.open(self.database) as archive:
            capture_id = self.capture_sample(archive, plan, 'retest_adaptive.jsonl')
            self.assertEqual(plan_retest(archive, capture_id, 1, 2, 3)['status'], 'no_guarantee')
            record = plan_retest(archive, capture_id, 1, 2, 4)
            self.assertEqual(record['worst_cost'], 4)
            self.assertEqual(record['tree'], ADAPTIVE_TREE_B4)
            self.assertEqual(record['tree']['branches']['0']['branches']['0'], {'value': [0, 0]})
            self.assertTrue(verify_retest(archive, record, capture_id, 1, 2, 4))

    def test_adaptive_sample_with_duplicate_mask_action(self):
        plan = PlanRegistry().register(
            'station-12', 3, {'x': 1, 'y': 2},
            actions={'A': (4, 2), 'B': (6, 1), 'C': (7, 1), 'D': (7, 1)},
        )
        with SqliteArchive.open(self.database) as archive:
            capture_id = self.capture_sample(archive, plan, 'retest_adaptive.jsonl')
            record = plan_retest(archive, capture_id, 1, 2, 6)
            self.assertEqual(record['worst_cost'], 4)
            self.assertEqual(record['tree'], ADAPTIVE_TREE_B4)


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.database = self.root / 'compat.sqlite3'
        self.recording = self.root / 'input.jsonl'

    def publish(self, archive, plan, frames, **options):
        path = write_recording(self.recording, frames, width=plan.bit_width)
        return capture_file(archive, plan, path, **options)

    def test_undeclared_actions_unavailable(self):
        plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2})
        with SqliteArchive.open(self.database) as archive:
            capture_id = self.publish(archive, plan, (('a', 1, 0), ('b', 1, 0)))
            record = plan_retest(archive, capture_id, 1, 2, 6)
            self.assertEqual(record, {
                'capture_id': capture_id, 'masks': [1, 2], 'budget': 6,
                'status': 'unavailable', 'worst_cost': None, 'tree': None,
            })
            with self.assertRaises(QueryParameterError):
                plan_retest(archive, capture_id, 0, 2, 6)
            with self.assertRaises(QueryParameterError):
                plan_retest(archive, capture_id, 1, 2, 7)
            self.assertTrue(verify_retest(archive, record, capture_id, 1, 2, 6))
        with SqliteArchive.open(self.database) as reopened:
            self.assertEqual(plan_retest(reopened, capture_id, 1, 2, 6), record)
            self.assertTrue(verify_retest(reopened, record, capture_id, 1, 2, 6))

    def test_legacy_payload_without_actions_field(self):
        plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2})
        with SqliteArchive.open(self.database) as archive:
            capture_id = self.publish(archive, plan, (('a', 1, 0), ('b', 1, 0)))
        legacy = json.dumps({
            'plan': {'key': 'station-12', 'bit_width': 2, 'loops': [['x', 1], ['y', 2]]},
            'frames': [[2, 'a', 1, 0], [3, 'b', 1, 0]],
        })
        with closing(sqlite3.connect(self.database)) as editor:
            editor.execute('UPDATE captures SET evidence = ? WHERE capture_id = ?', (legacy, capture_id))
            editor.commit()
        with SqliteArchive.open(self.database) as reopened:
            self.assertEqual(query(reopened, capture_id, 1), [(0,)])
            record = plan_retest(reopened, capture_id, 1, 2, 3)
            self.assertEqual(record['status'], 'unavailable')
            self.assertTrue(verify_retest(reopened, record, capture_id, 1, 2, 3))

    def test_explicit_empty_actions(self):
        registry = PlanRegistry()
        plan = registry.register('station-12', 2, {'x': 1, 'y': 2}, actions={})
        with SqliteArchive.open(self.database) as archive:
            unique = self.publish(archive, plan, (('a', 1, 0), ('b', 1, 0), ('c', 2, 1), ('d', 2, 1)))
            record = plan_retest(archive, unique, 1, 2, 0)
            self.assertEqual(record, {
                'capture_id': unique, 'masks': [1, 2], 'budget': 0,
                'status': 'ok', 'worst_cost': 0, 'tree': {'value': [0, 1]},
            })
            self.assertTrue(verify_retest(archive, record, unique, 1, 2, 0))
            ambiguous = self.publish(archive, plan, (('a', 1, 0), ('b', 1, 0)))
            record = plan_retest(archive, ambiguous, 1, 2, 6)
            self.assertEqual(record['status'], 'no_guarantee')
            self.assertIsNone(record['tree'])
            self.assertTrue(verify_retest(archive, record, ambiguous, 1, 2, 6))

    def test_actions_frozen_with_capture(self):
        registry = PlanRegistry()
        plan = registry.register('station-12', 2, {'x': 1, 'y': 2}, actions={'A': (2, 2), 'B': (3, 1)})
        with SqliteArchive.open(self.database) as archive:
            capture_id = self.publish(archive, plan, (('a', 1, 0), ('b', 1, 0)))
            other = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2}, actions={'B': (1, 1)})
            second = self.publish(archive, other, (('a', 1, 0), ('b', 1, 0)))
        with SqliteArchive.open(self.database) as reopened:
            frozen = reopened.get(capture_id)
            self.assertEqual(frozen.plan, plan)
            self.assertEqual(frozen.plan.actions.items(), (('A', (2, 2)), ('B', (3, 1))))
            self.assertEqual(reopened.get(second).plan.actions.items(), (('B', (1, 1)),))
            self.assertEqual(plan_retest(reopened, capture_id, 1, 2, 3)['worst_cost'], 3)
            self.assertEqual(plan_retest(reopened, second, 1, 2, 3)['status'], 'no_guarantee')

    def test_capture_failure_preserves_previous(self):
        plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2}, actions={'B': (3, 1)})
        candidates = []

        def interrupt(candidate_id):
            candidates.append(candidate_id)
            raise OSError('值班员停止提交')

        with SqliteArchive.open(self.database) as archive:
            previous = self.publish(archive, plan, (('a', 1, 0), ('b', 1, 0)))
            old_record = plan_retest(archive, previous, 1, 2, 3)
            with self.assertRaisesRegex(CaptureRejected, '值班员停止提交'):
                self.publish(archive, plan, (('c', 1, 0), ('d', 1, 0)), before_commit=interrupt)
            with self.assertRaises(CaptureNotFound):
                archive.get(candidates[0])
        with SqliteArchive.open(self.database) as reopened:
            self.assertEqual(plan_retest(reopened, previous, 1, 2, 3), old_record)
            self.assertTrue(verify_retest(reopened, old_record, previous, 1, 2, 3))
            with self.assertRaises(CaptureNotFound):
                reopened.get(candidates[0])

    def test_retest_materials_baseline(self):
        retest_materials.main()


class RetestParameterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.database = self.root / 'params.sqlite3'
        self.plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2}, actions={'B': (3, 1)})
        self.archive = SqliteArchive.open(self.database)
        self.addCleanup(self.archive.close)
        path = write_recording(self.root / 'input.jsonl', (('a', 1, 0), ('b', 1, 0)))
        self.capture_id = capture_file(self.archive, self.plan, path)

    def test_invalid_masks_and_budget(self):
        for masks in ((0, 2), (4, 2), (-1, 2), (1.0, 2), (True, 2), (None, 2), (1, 1 << 2)):
            with self.subTest(masks=masks):
                with self.assertRaises(QueryParameterError):
                    plan_retest(self.archive, self.capture_id, *masks, 3)
        for budget in (-1, 7, 1.0, True, None, '3'):
            with self.subTest(budget=budget):
                with self.assertRaises(QueryParameterError):
                    plan_retest(self.archive, self.capture_id, 1, 2, budget)
        with self.assertRaises(CaptureNotFound):
            plan_retest(self.archive, self.capture_id + 1, 1, 2, 3)

    def test_same_mask_target(self):
        record = plan_retest(self.archive, self.capture_id, 1, 1, 3)
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(record['tree'], {'value': [0, 0]})
        self.assertTrue(verify_retest(self.archive, record, self.capture_id, 1, 1, 3))

    def test_damaged_evidence(self):
        good = plan_retest(self.archive, self.capture_id, 1, 2, 3)
        with closing(sqlite3.connect(self.database)) as reader:
            original = reader.execute(
                'SELECT evidence FROM captures WHERE capture_id = ?', (self.capture_id,)
            ).fetchone()[0]
        payloads = (
            '{',
            json.dumps({
                'plan': {'key': 'station-12', 'bit_width': 2, 'loops': [['x', 1]], 'actions': [['B', 3]]},
                'frames': [[2, 'a', 1, 0]],
            }),
            json.dumps({
                'plan': {'key': 'station-12', 'bit_width': 2, 'loops': [['x', 1]], 'actions': [['B', 3, 9]]},
                'frames': [[2, 'a', 1, 0]],
            }),
            json.dumps({
                'plan': {'key': 'station-12', 'bit_width': 2, 'loops': [['x', 1]], 'actions': [['B', 3, 1]]},
                'frames': [[2, 'a', 1, 0], [3, 'b', 1, 0], [4, 'c', 1, 1], [5, 'd', 1, 1]],
            }),
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                with closing(sqlite3.connect(self.database)) as editor:
                    editor.execute(
                        'UPDATE captures SET evidence = ? WHERE capture_id = ?',
                        (payload, self.capture_id),
                    )
                    editor.commit()
                with self.assertRaises(ArchiveError):
                    plan_retest(self.archive, self.capture_id, 1, 2, 3)
                self.assertFalse(verify_retest(self.archive, good, self.capture_id, 1, 2, 3))
        with closing(sqlite3.connect(self.database)) as editor:
            editor.execute(
                'UPDATE captures SET evidence = ? WHERE capture_id = ?',
                (original, self.capture_id),
            )
            editor.commit()
        self.assertEqual(plan_retest(self.archive, self.capture_id, 1, 2, 3), good)


class VerifyRetestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.archive = SqliteArchive.open(self.root / 'verify.sqlite3')
        self.addCleanup(self.archive.close)
        plan = PlanRegistry().register('station-12', 2, {'x': 1, 'y': 2}, actions={'A': (2, 2), 'B': (3, 1)})
        path = write_recording(self.root / 'input.jsonl', (('a', 1, 0), ('b', 1, 0)))
        self.capture_id = capture_file(self.archive, plan, path)
        self.record = plan_retest(self.archive, self.capture_id, 1, 2, 3)

    def test_roundtrip_and_tampering(self):
        self.assertEqual(self.record['status'], 'ok')
        self.assertTrue(verify_retest(self.archive, self.record, self.capture_id, 1, 2, 3))
        decoded = json.loads(json.dumps(self.record))
        self.assertTrue(verify_retest(self.archive, decoded, self.capture_id, 1, 2, 3))
        tampered = []
        for field, value in (
            ('capture_id', self.capture_id + 1), ('masks', [2, 1]), ('budget', 2),
            ('status', 'no_guarantee'), ('worst_cost', 2), ('worst_cost', True), ('tree', None),
        ):
            changed = json.loads(json.dumps(self.record))
            changed[field] = value
            tampered.append((field, changed))
        leaf_changed = json.loads(json.dumps(self.record))
        leaf_changed['tree']['branches']['0']['branches']['0'] = {'value': [0, 1]}
        tampered.append(('leaf', leaf_changed))
        action_changed = json.loads(json.dumps(self.record))
        action_changed['tree']['action_id'] = 'A'
        tampered.append(('action', action_changed))
        extra = json.loads(json.dumps(self.record))
        extra['note'] = 1
        tampered.append(('extra', extra))
        missing = json.loads(json.dumps(self.record))
        del missing['budget']
        tampered.append(('missing', missing))
        for label, changed in tampered:
            with self.subTest(label=label):
                self.assertFalse(verify_retest(self.archive, changed, self.capture_id, 1, 2, 3))
        for bad_record in (None, [], 'record', 1, {'status': 'ok'}):
            with self.subTest(bad_record=bad_record):
                self.assertFalse(verify_retest(self.archive, bad_record, self.capture_id, 1, 2, 3))

    def test_invalid_request(self):
        self.assertFalse(verify_retest(self.archive, self.record, self.capture_id + 1, 1, 2, 3))
        self.assertFalse(verify_retest(self.archive, self.record, self.capture_id, 0, 2, 3))
        self.assertFalse(verify_retest(self.archive, self.record, self.capture_id, 1, 2, 9))
        self.assertFalse(verify_retest(self.archive, self.record, self.capture_id, 1, 2, 2))


class OracleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.counter = 0

    def check_configuration(self, width, frames, actions, targets, budgets):
        self.counter += 1
        key = f'oracle-{self.counter}'
        registry = PlanRegistry()
        plan = registry.register(key, width, {}, actions=actions)
        path = write_recording(
            self.root / 'oracle.jsonl',
            tuple((str(index), mask, value) for index, (mask, value) in enumerate(frames)),
            key=key,
            width=width,
        )
        with SqliteArchive.open(self.root / 'oracle.sqlite3') as archive:
            try:
                capture_id = capture_file(archive, plan, path)
            except CaptureRejected:
                possible = any(
                    sum(state_parity(state, mask) != value for mask, value in frames) <= 1
                    for state in range(1 << width)
                )
                self.assertFalse(possible, (width, frames))
                return
            for masks in targets:
                for budget in budgets:
                    with self.subTest(width=width, frames=frames, actions=actions, masks=masks, budget=budget):
                        record = plan_retest(archive, capture_id, masks[0], masks[1], budget)
                        expected = oracle_plan(width, frames, masks, actions, budget)
                        if expected is None:
                            self.assertEqual(record['status'], 'no_guarantee')
                            self.assertIsNone(record['worst_cost'])
                            self.assertIsNone(record['tree'])
                        else:
                            self.assertEqual(record['status'], 'ok')
                            self.assertEqual(record['worst_cost'], expected[0])
                            self.assertEqual(record['tree'], expected[1])
                        self.assertTrue(
                            verify_retest(archive, record, capture_id, masks[0], masks[1], budget)
                        )
            (self.root / 'oracle.sqlite3').unlink()

    def test_width_one_exhaustive(self):
        action_sets = (
            {},
            {'A': (1, 1)},
            {'A': (1, 2), 'B': (1, 1)},
        )
        frames_pool = [(1, 0), (1, 1)]
        configs = [()]
        for length in (1, 2, 3):
            configs.extend(product(frames_pool, repeat=length))
        for actions in action_sets:
            for frames in configs:
                self.check_configuration(1, frames, actions, [(1, 1)], range(7))

    def test_width_two_exhaustive(self):
        action_sets = (
            {},
            {'A': (1, 1)},
            {'A': (2, 2), 'B': (3, 1)},
            {'A': (3, 1), 'B': (1, 2), 'C': (2, 1)},
        )
        frames_pool = [(mask, value) for mask in (1, 2, 3) for value in (0, 1)]
        configs = [()]
        for length in (1, 2):
            configs.extend(product(frames_pool, repeat=length))
        targets = [(1, 1), (1, 2), (2, 1), (3, 3)]
        for actions in action_sets:
            for frames in configs:
                self.check_configuration(2, frames, actions, targets, range(7))

    def test_width_three_spot(self):
        action_sets = (
            {'A': (4, 2), 'B': (6, 1), 'C': (7, 1)},
            {'A': (5, 1), 'B': (3, 1), 'C': (6, 2), 'D': (7, 1)},
        )
        configs = (
            ((4, 0), (1, 0), (2, 0)),
            ((7, 1), (3, 0), (5, 1)),
            ((1, 1), (2, 1), (4, 1), (7, 0)),
        )
        targets = [(1, 2), (4, 1), (7, 7)]
        for actions in action_sets:
            for frames in configs:
                self.check_configuration(3, frames, actions, targets, range(7))


class HighBitTests(unittest.TestCase):
    def test_high_bit_actions(self):
        high = 1 << 63
        plan = PlanRegistry().register('wide', 64, {'highest': high}, actions={'A': (high, 6), 'B': (1, 1)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_recording(root / 'input.jsonl', (('a', high, 1), ('b', high, 1)), key='wide', width=64)
            with SqliteArchive.open(root / 'wide.sqlite3') as archive:
                capture_id = capture_file(archive, plan, path)
                record = plan_retest(archive, capture_id, high, 1, 6)
                self.assertEqual(record['status'], 'ok')
                self.assertEqual(record['worst_cost'], 3)
                self.assertEqual(record['tree']['action_id'], 'B')
                self.assertTrue(verify_retest(archive, record, capture_id, high, 1, 6))
                unique = plan_retest(archive, capture_id, high, high, 6)
                self.assertEqual(unique['tree'], {'value': [1, 1]})
                self.assertEqual(unique['worst_cost'], 0)
