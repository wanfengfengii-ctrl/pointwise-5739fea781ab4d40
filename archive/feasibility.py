from collections.abc import Iterable
from itertools import chain

from admission.evidence import CaptureEvidence
from archive.linear import Equation, equations_consistent


class NoExplanation(ValueError):
    """整批观测无法由至多一条读值失配解释。"""


def observations_explainable(observations: Iterable[Equation], hard: Iterable[Equation] = ()) -> bool:
    """在全部硬条件成立时，判断观测序列是否存在至多一次失配。"""
    required = tuple(hard)
    observations = tuple(observations)
    if equations_consistent(chain(required, observations)):
        return True

    # 每次省略一个物理观测位置，硬条件始终参与消元。
    for index in range(len(observations)):
        kept = chain(required, observations[:index], observations[index + 1:])
        if equations_consistent(kept):
            return True
    return False


def has_explanation(evidence: CaptureEvidence, hard: Iterable[Equation] = ()) -> bool:
    """在所有硬条件成立时，判断是否存在全批至多一次失配。"""
    observations = tuple((frame.mask, frame.value) for frame in evidence.frames)
    return observations_explainable(observations, hard)
