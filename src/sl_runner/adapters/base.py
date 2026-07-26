"""Backend-neutral release adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    run_id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedPrompt:
    run_id: str
    input_token_ids: list[int]
    prompt_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    run_id: str
    text: str
    input_token_ids: list[int]
    output_token_ids: list[int]
    input_token_hash: str
    output_token_hash: str
    finish_reason: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedIdentity:
    metadata: dict[str, Any]


class ModelAdapter(ABC):
    @abstractmethod
    def load(self) -> LoadedIdentity: ...

    @abstractmethod
    def render_prompt(self, request: GenerationRequest) -> RenderedPrompt: ...

    @abstractmethod
    def generate(self, batch: list[RenderedPrompt]) -> list[GenerationResult]: ...

    @abstractmethod
    def capture_metadata(self) -> dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...

