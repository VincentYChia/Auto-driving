"""NAVSIM agent that replays trajectories predicted offline (run in the NAVSIM env).

MVLAD-AD and NAVSIM need incompatible Python environments, so the model runs separately
(``navsim_eval/predict_mvlad.py``) and this agent feeds its saved waypoints to the PDM scorer.
Predictions were made from AgentInput only; ``requires_scene`` is set solely to read the
scene token that keys the prediction file.

Use it by overriding NAVSIM's default agent (works for v1.1 and v2):

    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py train_test_split=navtest \
        experiment_name=mvlad_navtest \
        agent._target_=navsim_eval.precomputed_agent.PrecomputedTrajectoryAgent \
        +agent.predictions_path=$NAVSIM_EXP_ROOT/mvlad/predictions
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory

from navsim_eval.predictions import load_predictions, poses_from_waypoints

logger = logging.getLogger(__name__)


class PrecomputedTrajectoryAgent(AbstractAgent):
    """Returns stored waypoints (first ``num_poses`` at 0.5 s) with headings from the direction of travel."""

    requires_scene = True

    def __init__(
        self,
        predictions_path: str,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        fallback: Optional[str] = None,
    ):
        try:  # NAVSIM v2 signature
            super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        except TypeError:  # NAVSIM v1.1 signature
            super().__init__(requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self._predictions_path = predictions_path
        self._fallback = fallback  # None: fail on a missing token; "constant_velocity": substitute CV
        self._predictions: Dict[str, np.ndarray] = {}

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._predictions = load_predictions(self._predictions_path)
        logger.info("loaded %d precomputed trajectories from %s", len(self._predictions), self._predictions_path)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Optional[Scene] = None) -> Trajectory:
        num_poses = self._trajectory_sampling.num_poses
        token = scene.scene_metadata.initial_token if scene is not None else None
        waypoints = self._predictions.get(token)
        if waypoints is None:
            if self._fallback != "constant_velocity":
                raise KeyError(f"no precomputed trajectory for token {token}")
            speed = float(np.hypot(*agent_input.ego_statuses[-1].ego_velocity[:2]))
            dt = self._trajectory_sampling.interval_length
            waypoints = np.array([[(i + 1) * dt * speed, 0.0] for i in range(num_poses)])
            logger.warning("token %s missing; using constant velocity", token)
        return Trajectory(poses_from_waypoints(waypoints, num_poses), self._trajectory_sampling)
