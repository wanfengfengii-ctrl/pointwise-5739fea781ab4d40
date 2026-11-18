from collections.abc import Mapping

from plans.actions import ActionCatalog
from plans.catalog import LoopCatalog
from plans.model import PlanConflict, SamplingPlan


class PlanRegistry:
    def __init__(self) -> None:
        self._plans: dict[str, SamplingPlan] = {}

    def register(
        self,
        key: str | None = None,
        bit_width: int | None = None,
        loops: Mapping[str, int] | None = None,
        *,
        actions: Mapping[str, object] | None = None,
    ) -> SamplingPlan:
        try:
            catalog = LoopCatalog.from_mapping(bit_width, loops)
            action_catalog = None if actions is None else ActionCatalog.from_mapping(bit_width, actions)
        except ValueError as exc:
            raise PlanConflict(str(exc)) from exc
        candidate = SamplingPlan(key, bit_width, catalog, action_catalog)
        registered = self._plans.get(candidate.key)
        if registered is None:
            self._plans[candidate.key] = candidate
            return candidate
        if registered != candidate:
            raise PlanConflict(f'采样键 {candidate.key!r} 已登记了不同声明')
        return registered

    def get(self, key: str) -> SamplingPlan:
        return self._plans[key]
