from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common.replay import ReplayState


class Predictor(ABC):
    name: str
    description: str

    @abstractmethod
    def fit(self, train_runs: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: ReplayState, summary: dict[str, Any]) -> tuple[str, float]:
        raise NotImplementedError

    def predict_time_conditions(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, bool]] | None:
        return None

    def predict_time_condition_scores(
        self,
        state: ReplayState,
        summary: dict[str, Any],
        horizons: list[float],
    ) -> dict[str, dict[str, float]] | None:
        return None

    def save_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement checkpoint saving")
