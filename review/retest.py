"""按固定成果生成补采指令，并在下一班凭归档与约定参数复核同一份指令。"""

from itertools import product

from archive.artifacts import CaptureNotFound  # noqa: F401  (调用方沿用的错误类型)
from archive.feasibility import observations_explainable
from archive.storage import ArchiveError, SqliteArchive  # noqa: F401
from plans.masks import validate_mask
from review.parameters import QueryParameterError

MAX_BUDGET = 6

_STATUS_OK = 'ok'
_STATUS_NO_GUARANTEE = 'no_guarantee'
_STATUS_UNAVAILABLE = 'unavailable'

_RECORD_FIELDS = frozenset(('capture_id', 'masks', 'budget', 'status', 'worst_cost', 'tree'))


def _joint_values(observations: tuple[tuple[int, int], ...], masks: tuple[int, int]) -> list[tuple[int, int]]:
    """列出当前观测下仍可由同一源状态支撑的联合值，按字典序。"""
    values = []
    for candidate in product((0, 1), repeat=2):
        hard = tuple(zip(masks, candidate, strict=True))
        if observations_explainable(observations, hard):
            values.append(candidate)
    return values


class _Planner:
    """在一条补采路径上共享一次失配额度的指令搜索。"""

    def __init__(
        self,
        base: tuple[tuple[int, int], ...],
        masks: tuple[int, int],
        actions: tuple[tuple[str, int, int], ...],
    ) -> None:
        self._base = base
        self._masks = masks
        self._actions = actions
        self._memo: dict[tuple[tuple[tuple[int, int], ...], int], tuple[int, dict] | None] = {}

    def solve(self, budget: int) -> tuple[int, dict] | None:
        return self._solve((), budget)

    def _solve(self, extras: tuple[tuple[int, int], ...], budget: int) -> tuple[int, dict] | None:
        key = (extras, budget)
        if key in self._memo:
            return self._memo[key]
        joints = _joint_values(self._base + extras, self._masks)
        if not joints:
            result = None
        elif len(joints) == 1:
            result = (0, {'value': [joints[0][0], joints[0][1]]})
        else:
            result = self._best_action(extras, budget)
        self._memo[key] = result
        return result

    def _best_action(self, extras: tuple[tuple[int, int], ...], budget: int) -> tuple[int, dict] | None:
        best = None
        for action_id, mask, cost in self._actions:
            if cost > budget:
                continue
            branches = {}
            worst_child = 0
            covered = True
            for reading in (0, 1):
                child_extras = tuple(sorted(extras + ((mask, reading),)))
                if not observations_explainable(self._base + child_extras):
                    continue
                child = self._solve(child_extras, budget - cost)
                if child is None:
                    covered = False
                    break
                branches[str(reading)] = child[1]
                worst_child = max(worst_child, child[0])
            if not covered or not branches:
                continue
            total = cost + worst_child
            if best is None or total < best[0]:
                best = (total, {'action_id': action_id, 'branches': branches})
        return best


def plan_retest(
    archive: SqliteArchive,
    capture_id: int,
    mask1: int,
    mask2: int,
    budget: int,
) -> dict[str, object]:
    """按固定成果生成补采指令记录，可直接往返 JSON。"""
    frozen = archive.get(capture_id)
    width = frozen.plan.bit_width
    try:
        masks = (validate_mask(mask1, width), validate_mask(mask2, width))
    except ValueError as exc:
        raise QueryParameterError(str(exc)) from exc
    if type(budget) is not int or not 0 <= budget <= MAX_BUDGET:
        raise QueryParameterError(f'预算必须是 0～{MAX_BUDGET} 的整数')

    record: dict[str, object] = {
        'capture_id': frozen.capture_id,
        'masks': [masks[0], masks[1]],
        'budget': budget,
    }
    actions = frozen.plan.actions
    if actions is None:
        record.update(status=_STATUS_UNAVAILABLE, worst_cost=None, tree=None)
        return record

    base = tuple((frame.mask, frame.value) for frame in frozen.evidence.frames)
    if not _joint_values(base, masks):
        raise ArchiveError(f'成果 {frozen.capture_id} 的证据没有可解释状态')
    ordered = tuple((action_id, mask, cost) for action_id, (mask, cost) in actions.items())
    solved = _Planner(base, masks, ordered).solve(budget)
    if solved is None:
        record.update(status=_STATUS_NO_GUARANTEE, worst_cost=None, tree=None)
    else:
        worst_cost, tree = solved
        record.update(status=_STATUS_OK, worst_cost=worst_cost, tree=tree)
    return record


def _valid_node(node: object) -> bool:
    if type(node) is not dict:
        return False
    keys = set(node)
    if keys == {'value'}:
        value = node['value']
        return (
            type(value) is list
            and len(value) == 2
            and all(type(bit) is int and bit in (0, 1) for bit in value)
        )
    if keys == {'action_id', 'branches'}:
        if type(node['action_id']) is not str or not node['action_id']:
            return False
        branches = node['branches']
        if type(branches) is not dict or not branches:
            return False
        if any(type(key) is not str or key not in ('0', '1') for key in branches):
            return False
        return all(_valid_node(child) for child in branches.values())
    return False


def _valid_record(record: object) -> bool:
    if type(record) is not dict or set(record) != _RECORD_FIELDS:
        return False
    if type(record['capture_id']) is not int:
        return False
    masks = record['masks']
    if type(masks) is not list or len(masks) != 2 or any(type(mask) is not int for mask in masks):
        return False
    if type(record['budget']) is not int:
        return False
    status = record['status']
    if status == _STATUS_OK:
        return type(record['worst_cost']) is int and _valid_node(record['tree'])
    if status in (_STATUS_NO_GUARANTEE, _STATUS_UNAVAILABLE):
        return record['worst_cost'] is None and record['tree'] is None
    return False


def verify_retest(
    archive: SqliteArchive,
    record: object,
    expected_capture_id: int,
    expected_mask1: int,
    expected_mask2: int,
    expected_budget: int,
) -> bool:
    """按调用者约定从归档重算，完整核对整份补采指令记录。"""
    try:
        expected = plan_retest(
            archive,
            expected_capture_id,
            expected_mask1,
            expected_mask2,
            expected_budget,
        )
    except (ArchiveError, CaptureNotFound, QueryParameterError):
        return False
    if not _valid_record(record):
        return False
    return record == expected
