from collections.abc import Mapping
from dataclasses import dataclass

from plans.masks import validate_bit_width, validate_mask

MAX_ACTIONS = 4
MIN_ACTION_COST = 1
MAX_ACTION_COST = 6


def _validate_spec(spec: object) -> tuple[int, int]:
    if type(spec) not in (tuple, list) or len(spec) != 2:
        raise ValueError('测法定义必须是 (掩码, 成本) 二元组')
    mask, cost = spec
    if type(cost) is not int or not MIN_ACTION_COST <= cost <= MAX_ACTION_COST:
        raise ValueError(f'测法成本必须是 {MIN_ACTION_COST}～{MAX_ACTION_COST} 的整数')
    return mask, cost


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    """一次声明内登记的补采测法表，按测法 ID 排序冻结。"""

    bit_width: int
    _entries: tuple[tuple[str, tuple[int, int]], ...]

    def __post_init__(self) -> None:
        width = validate_bit_width(self.bit_width)
        by_id: dict[str, tuple[int, int]] = {}
        for action_id, spec in self._entries:
            if type(action_id) is not str or not action_id:
                raise ValueError('测法 ID 必须是非空文本')
            if action_id in by_id:
                raise ValueError(f'重复的测法 ID {action_id!r}')
            mask, cost = _validate_spec(spec)
            by_id[action_id] = (validate_mask(mask, width), cost)
        if len(by_id) > MAX_ACTIONS:
            raise ValueError(f'测法至多登记 {MAX_ACTIONS} 项')
        object.__setattr__(self, '_entries', tuple(sorted(by_id.items())))

    @classmethod
    def from_mapping(cls, bit_width: int, actions: Mapping[str, object]) -> 'ActionCatalog':
        if not isinstance(actions, Mapping):
            raise ValueError('测法表必须是从测法 ID 到 (掩码, 成本) 的映射')
        return cls(bit_width, tuple(actions.items()))

    def get(self, action_id: str) -> tuple[int, int]:
        for known_id, spec in self._entries:
            if action_id == known_id:
                return spec
        raise KeyError(action_id)

    def items(self) -> tuple[tuple[str, tuple[int, int]], ...]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)
