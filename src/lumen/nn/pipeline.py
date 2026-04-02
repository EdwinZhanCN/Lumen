"""MLPipeline implementation for LumNN's declarative execution graph."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .context import MLContext
from .exceptions import (
    InvalidMLPacketError,
    InvalidPipelineGraphError,
    NodeExecutionError,
    PipelineNotCompiledError,
)
from .node import MLNode
from .packet import MLPacket


@dataclass(frozen=True, slots=True)
class _NodeStep:
    """Internal compiled representation for a pipeline node."""

    node: MLNode
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


class MLPipeline:
    """Declarative pipeline that builds and executes a LumNN DAG.

    Args:
        context: Runtime context used during compilation and dispatch.
    """

    def __init__(self, context: MLContext) -> None:
        self._context = context
        self._pipeline_id = id(self)
        self._inputs: dict[str, MLPacket] = {}
        self._packets: dict[str, MLPacket] = {}
        self._steps: list[_NodeStep] = []
        self._compiled = False
        self._compiled_step_indices: tuple[int, ...] = ()
        self._compiled_outputs: tuple[str, ...] = ()
        self._compiled_inputs: tuple[str, ...] = ()

    def input(
        self,
        name: str,
        *,
        dtype: str | None = None,
        mime: str | None = None,
        device: str = "host",
    ) -> MLPacket:
        """Declare a symbolic input packet for this pipeline."""
        self._ensure_mutable()
        self._ensure_unique_packet_name(name)
        packet = MLPacket(
            name=name,
            dtype=dtype,
            device=self._context.resolve_device(device),
            mime=mime,
            metadata={"kind": "pipeline_input"},
            _pipeline_id=self._pipeline_id,
            _source_kind="input",
        )
        self._inputs[name] = packet
        self._packets[name] = packet
        return packet

    def add(
        self,
        node: MLNode,
        *,
        inputs: list[MLPacket],
        outputs: list[str] | None = None,
    ) -> MLPacket | list[MLPacket]:
        """Attach a node to the graph and declare its output packets."""
        self._ensure_mutable()
        self._validate_pipeline_packets(inputs)

        output_names = tuple(outputs or node.output_names)
        if not output_names:
            raise InvalidPipelineGraphError(
                f"Node '{node.name}' must declare at least one output."
            )
        if outputs is not None and len(outputs) != len(node.output_names):
            raise InvalidPipelineGraphError(
                f"Node '{node.name}' expected {len(node.output_names)} outputs, "
                f"got {len(outputs)}."
            )

        for output_name in output_names:
            self._ensure_unique_packet_name(output_name)

        step = _NodeStep(
            node=node,
            input_names=tuple(packet.name for packet in inputs),
            output_names=output_names,
        )
        self._steps.append(step)

        produced_packets = [
            MLPacket(
                name=output_name,
                dtype=None,
                device=self._context.resolve_device(node.device_hint),
                metadata={"kind": "node_output"},
                _pipeline_id=self._pipeline_id,
                _source_kind="node",
                _source_node=node.name,
            )
            for output_name in output_names
        ]
        for packet in produced_packets:
            self._packets[packet.name] = packet

        if len(produced_packets) == 1:
            return produced_packets[0]
        return produced_packets

    def compile(self, *, outputs: list[MLPacket]) -> MLPipeline:
        """Validate the DAG and freeze the pipeline for execution."""
        self._ensure_mutable()
        self._validate_pipeline_packets(outputs)
        if not outputs:
            raise InvalidPipelineGraphError("Pipeline must declare at least one output.")

        producer_by_packet: dict[str, int] = {}
        for index, step in enumerate(self._steps):
            for output_name in step.output_names:
                if output_name in producer_by_packet:
                    raise InvalidPipelineGraphError(
                        f"Packet '{output_name}' is produced by multiple nodes."
                    )
                producer_by_packet[output_name] = index

        reachable_steps = self._collect_reachable_steps(
            outputs=outputs,
            producer_by_packet=producer_by_packet,
        )
        ordered_steps = self._topological_sort(
            reachable_steps=reachable_steps,
            producer_by_packet=producer_by_packet,
        )

        required_inputs: list[str] = []
        seen_inputs: set[str] = set()
        for step_index in ordered_steps:
            step = self._steps[step_index]
            for input_name in step.input_names:
                if input_name in self._inputs and input_name not in seen_inputs:
                    required_inputs.append(input_name)
                    seen_inputs.add(input_name)

        for packet in outputs:
            if packet.name in self._inputs and packet.name not in seen_inputs:
                required_inputs.append(packet.name)
                seen_inputs.add(packet.name)

        self._compiled = True
        self._compiled_step_indices = tuple(ordered_steps)
        self._compiled_outputs = tuple(packet.name for packet in outputs)
        self._compiled_inputs = tuple(required_inputs)
        return self

    def dispatch(self, inputs: dict[str, object]) -> dict[str, MLPacket]:
        """Execute the compiled graph for a concrete set of inputs."""
        if not self._compiled:
            raise PipelineNotCompiledError(
                "Pipeline must be compiled before dispatch()."
            )

        provided_names = set(inputs)
        required_names = set(self._compiled_inputs)
        missing = sorted(required_names - provided_names)
        if missing:
            raise InvalidMLPacketError(
                f"Missing pipeline inputs for dispatch: {missing}"
            )
        unexpected = sorted(provided_names - required_names)
        if unexpected:
            raise InvalidMLPacketError(
                f"Unexpected pipeline inputs for dispatch: {unexpected}"
            )

        packet_values: dict[str, MLPacket] = {}
        for input_name in self._compiled_inputs:
            symbolic = self._inputs[input_name]
            packet_values[input_name] = self._materialize_input_packet(
                name=input_name,
                symbolic=symbolic,
                value=inputs[input_name],
            )

        for step_index in self._compiled_step_indices:
            step = self._steps[step_index]
            step_inputs = {
                input_name: packet_values[input_name] for input_name in step.input_names
            }
            if step.node.device_hint:
                target_device = self._context.resolve_device(step.node.device_hint)
                step_inputs = {
                    input_name: (
                        packet
                        if packet.device == target_device
                        else self._context.transfer(packet, target_device)
                    )
                    for input_name, packet in step_inputs.items()
                }

            step.node.initialize(self._context)
            try:
                raw_outputs = step.node.run(self._context, step_inputs)
            except Exception as exc:
                raise NodeExecutionError(step.node.name, str(exc)) from exc

            expected = set(step.output_names)
            actual = set(raw_outputs)
            if actual != expected:
                raise NodeExecutionError(
                    step.node.name,
                    f"Expected outputs {sorted(expected)}, got {sorted(actual)}.",
                )

            for output_name in step.output_names:
                output_packet = raw_outputs[output_name]
                if not isinstance(output_packet, MLPacket):
                    raise NodeExecutionError(
                        step.node.name,
                        f"Output '{output_name}' is not an MLPacket instance.",
                    )
                packet_values[output_name] = replace(output_packet, name=output_name)

        return {
            output_name: packet_values[output_name]
            for output_name in self._compiled_outputs
        }

    def _ensure_mutable(self) -> None:
        """Ensure the graph is still mutable."""
        if self._compiled:
            raise InvalidPipelineGraphError(
                "Pipeline has been compiled and can no longer be modified."
            )

    def _ensure_unique_packet_name(self, name: str) -> None:
        """Ensure packet names stay unique inside one pipeline."""
        if name in self._packets:
            raise InvalidPipelineGraphError(
                f"Packet '{name}' is already declared in this pipeline."
            )

    def _validate_pipeline_packets(self, packets: Iterable[MLPacket]) -> None:
        """Validate that packets belong to this pipeline."""
        for packet in packets:
            if packet._pipeline_id != self._pipeline_id:
                raise InvalidPipelineGraphError(
                    f"Packet '{packet.name}' does not belong to this pipeline."
                )

    def _collect_reachable_steps(
        self,
        *,
        outputs: list[MLPacket],
        producer_by_packet: dict[str, int],
    ) -> set[int]:
        """Collect all node steps needed to materialize the declared outputs."""
        reachable: set[int] = set()
        pending = [packet.name for packet in outputs]

        while pending:
            packet_name = pending.pop()
            if packet_name in self._inputs:
                continue
            producer_index = producer_by_packet.get(packet_name)
            if producer_index is None:
                raise InvalidPipelineGraphError(
                    f"Packet '{packet_name}' has no producer in this pipeline."
                )
            if producer_index in reachable:
                continue
            reachable.add(producer_index)
            pending.extend(self._steps[producer_index].input_names)

        return reachable

    def _topological_sort(
        self,
        *,
        reachable_steps: set[int],
        producer_by_packet: dict[str, int],
    ) -> list[int]:
        """Return a topological execution order for the reachable node steps."""
        adjacency: dict[int, set[int]] = {index: set() for index in reachable_steps}
        in_degree: dict[int, int] = {index: 0 for index in reachable_steps}

        for step_index in reachable_steps:
            step = self._steps[step_index]
            for input_name in step.input_names:
                if input_name in self._inputs:
                    continue
                parent_index = producer_by_packet.get(input_name)
                if parent_index is None:
                    raise InvalidPipelineGraphError(
                        f"Packet '{input_name}' has no producer in this pipeline."
                    )
                if parent_index in reachable_steps and step_index not in adjacency[parent_index]:
                    adjacency[parent_index].add(step_index)
                    in_degree[step_index] += 1

        ready = deque(
            index
            for index in range(len(self._steps))
            if index in reachable_steps and in_degree[index] == 0
        )
        ordered: list[int] = []

        while ready:
            current = ready.popleft()
            ordered.append(current)
            for child in sorted(adjacency[current]):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)

        if len(ordered) != len(reachable_steps):
            raise InvalidPipelineGraphError(
                "Pipeline graph contains a cycle and cannot be compiled."
            )
        return ordered

    def _materialize_input_packet(
        self,
        *,
        name: str,
        symbolic: MLPacket,
        value: object,
    ) -> MLPacket:
        """Convert a dispatch input into a runtime packet."""
        if isinstance(value, MLPacket):
            return self._context.create_packet(
                name=name,
                value=value,
                dtype=symbolic.dtype,
                shape=symbolic.shape,
                device=symbolic.device,
                mime=symbolic.mime,
                metadata=dict(symbolic.metadata) | dict(value.metadata),
            )
        return self._context.create_packet(
            name=name,
            value=value,
            dtype=symbolic.dtype,
            shape=symbolic.shape,
            device=symbolic.device,
            mime=symbolic.mime,
            metadata=symbolic.metadata,
        )
