"""Timestamp-based replay of human observations."""

from __future__ import annotations

import bisect

from comotion_x.core.models import PoseFrame
from comotion_x.human_model.scenarios import HumanScenario


class HumanReplay:
    def __init__(self, scenario: HumanScenario, *, observations: bool = True) -> None:
        self.frames = (
            scenario.observation_frames if observations else scenario.ground_truth_frames
        )
        self._timestamps = tuple(frame.timestamp for frame in self.frames)

    def frame_at(self, timestamp: float) -> PoseFrame:
        if timestamp < 0:
            raise ValueError("replay time must be non-negative")
        index = bisect.bisect_right(self._timestamps, timestamp) - 1
        return self.frames[max(0, min(index, len(self.frames) - 1))]

