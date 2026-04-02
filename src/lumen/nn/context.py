"""MLContext runtime utilities for the LumNN orchestration layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from .exceptions import DeviceResolutionError
from .packet import MLPacket


class MLContext:
    """Runtime context for device resolution, packet creation, and cache reuse.

    Args:
        preferred_device: Preferred logical execution target. ``"auto"`` falls
            back to CPU when no other concrete device is available.
        execution_providers: Optional execution provider names associated with
            this context.
        available_devices: Optional explicit list of available devices.
    """

    def __init__(
        self,
        preferred_device: str = "auto",
        execution_providers: Iterable[str] | None = None,
        available_devices: Iterable[str] | None = None,
    ) -> None:
        self.preferred_device = preferred_device
        self.execution_providers = tuple(execution_providers or ())
        available = {
            self._normalize_device_name(device)
            for device in available_devices or ("host", "cpu")
        }
        available.add("host")
        available.add("cpu")
        self.available_devices = frozenset(available)
        self._session_cache: dict[object, object] = {}
        self._closed = False

    def create_packet(
        self,
        name: str,
        value: Any = None,
        *,
        dtype: str | None = None,
        shape: tuple[int, ...] | None = None,
        device: str | None = None,
        mime: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MLPacket:
        """Create a normalized packet bound to this context."""
        packet_device = self.resolve_device(device)
        if isinstance(value, MLPacket):
            return replace(
                value,
                name=name,
                dtype=dtype or value.dtype,
                shape=shape or value.shape,
                device=packet_device,
                mime=mime or value.mime,
                metadata=dict(value.metadata) | dict(metadata or {}),
            )
        return MLPacket(
            name=name,
            value=value,
            dtype=dtype,
            shape=shape,
            device=packet_device,
            mime=mime,
            metadata=metadata or {},
        )

    def resolve_device(self, preference: str | None = None) -> str:
        """Resolve a device hint into an available logical device."""
        raw = preference or self.preferred_device
        if raw in {"", "auto", None}:
            raw = self.preferred_device if self.preferred_device != "auto" else "cpu"

        normalized = self._normalize_device_name(raw)
        if normalized in self.available_devices:
            return normalized
        if normalized == "cpu":
            return "cpu"
        return "cpu"

    def get_session(
        self,
        key: object,
        factory: Callable[[], object] | None = None,
    ) -> object:
        """Return a cached session object, creating it on first use."""
        if key not in self._session_cache:
            if factory is None:
                raise KeyError(f"No cached session for key: {key!r}")
            self._session_cache[key] = factory()
        return self._session_cache[key]

    def transfer(self, packet: MLPacket, target_device: str) -> MLPacket:
        """Return a packet logically moved to another device."""
        resolved = self.resolve_device(target_device)
        return replace(
            packet,
            device=resolved,
            metadata=dict(packet.metadata)
            | {"transfer": "best_effort", "source_device": packet.device},
        )

    def close(self) -> None:
        """Close cached resources owned by this context."""
        if self._closed:
            return
        for session in self._session_cache.values():
            close = getattr(session, "close", None)
            if callable(close):
                close()
        self._session_cache.clear()
        self._closed = True

    @staticmethod
    def _normalize_device_name(device: str) -> str:
        """Normalize common device aliases into logical LumNN device strings."""
        normalized = device.strip().lower()
        if not normalized:
            raise DeviceResolutionError("Device hint must be a non-empty string.")
        aliases = {
            "host": "host",
            "cpu": "cpu",
            "gpu": "cuda:0",
            "cuda": "cuda:0",
            "mps": "mps",
            "coreml": "coreml",
            "openvino": "openvino",
            "dml": "dml",
            "directml": "dml",
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized.startswith(("cuda:", "mps:", "npu:", "openvino:", "dml:")):
            return normalized
        raise DeviceResolutionError(f"Unsupported device hint: {device}")
