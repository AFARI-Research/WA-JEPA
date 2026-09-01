"""Traffic policy for legacy NAVSIM navtest metric caches without agent tracks."""

from __future__ import annotations

from typing import Any, List

import numpy.typing as npt
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    AbstractTrafficAgentsPolicy,
)


class EmptyTrafficAgentsPolicy(AbstractTrafficAgentsPolicy):
    """Return an empty background scene for caches without tracked objects."""

    def __init__(self, future_trajectory_sampling: Any) -> None:
        self.future_trajectory_sampling = future_trajectory_sampling

    def get_list_of_simulated_object_types(self) -> List[TrackedObjectType]:
        return []

    def simulate_environment(
        self,
        simulated_ego_states: npt.NDArray,
        metric_cache: Any,
    ) -> List[DetectionsTracks]:
        return [
            DetectionsTracks(TrackedObjects([]))
            for _ in range(simulated_ego_states.shape[0])
        ]

    def simulate_traffic_agents(
        self,
        simulated_ego_states: npt.NDArray,
        metric_cache: Any,
    ) -> List[DetectionsTracks]:
        return [
            DetectionsTracks(TrackedObjects([]))
            for _ in range(max(int(simulated_ego_states.shape[0]) - 1, 0))
        ]
