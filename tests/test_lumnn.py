"""Tests for the LumNN orchestration framework."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lumen.nn import MLContext, MLPipeline, MLPacket
from lumen.nn._testing import (
    FailingNode,
    JoinTextNode,
    NumpyTensorNode,
    PassthroughNode,
    UppercaseTextNode,
)
from lumen.nn.exceptions import (
    InvalidMLPacketError,
    InvalidPipelineGraphError,
    NodeExecutionError,
    PipelineNotCompiledError,
)


def _only_packet(packet_or_packets: MLPacket | list[MLPacket]) -> MLPacket:
    """Return the only packet from an API call that should yield one output."""
    assert isinstance(packet_or_packets, MLPacket)
    return packet_or_packets


def test_mlpacket_inferrs_metadata_and_transfer_returns_new_packet() -> None:
    packet = MLPacket(
        name="image",
        value=b"abc",
        device="host",
        mime="image/png",
        metadata={"kind": "raw"},
    )
    ctx = MLContext(available_devices=("host", "cpu", "cuda:0"))

    moved = ctx.transfer(packet, "cuda")

    assert packet.dtype == "bytes"
    assert packet.shape == (3,)
    assert moved is not packet
    assert moved.device == "cuda:0"
    assert packet.device == "host"
    with pytest.raises(TypeError):
        packet.metadata["new"] = "value"  # type: ignore[index]


def test_pipeline_dispatch_runs_single_chain_and_initializes_node_once() -> None:
    ctx = MLContext()
    pipeline = MLPipeline(context=ctx)
    text = pipeline.input("text", dtype="string", device="host")
    uppercase = UppercaseTextNode()
    output = _only_packet(pipeline.add(uppercase, inputs=[text]))
    pipeline.compile(outputs=[output])

    first = pipeline.dispatch({"text": "lumenn"})
    second = pipeline.dispatch({"text": "graph"})

    assert first["upper_text"].value == "LUMENN"
    assert second["upper_text"].value == "GRAPH"
    assert uppercase.initialize_calls == 1


def test_pipeline_dispatch_supports_multi_input_chain_and_device_transfer() -> None:
    ctx = MLContext(available_devices=("host", "cpu", "cuda:0"))
    pipeline = MLPipeline(context=ctx)
    left = pipeline.input("left", dtype="string")
    right = pipeline.input("right", dtype="string")
    values = pipeline.input("values", dtype="float32")

    joined = _only_packet(
        pipeline.add(JoinTextNode(separator="-"), inputs=[left, right])
    )
    upper = _only_packet(
        pipeline.add(
            UppercaseTextNode(
                input_names=("joined",),
                output_names=("joined_upper",),
            ),
            inputs=[joined],
        )
    )
    tensor = _only_packet(
        pipeline.add(NumpyTensorNode(input_names=("values",)), inputs=[values])
    )
    pipeline.compile(outputs=[upper, tensor])

    result = pipeline.dispatch(
        {"left": "lumen", "right": "nn", "values": [1, 2, 3]}
    )

    assert result["joined_upper"].value == "LUMEN-NN"
    assert result["tensor"].device == "cuda:0"
    np.testing.assert_array_equal(
        result["tensor"].value,
        np.asarray([1, 2, 3], dtype=np.float32),
    )


def test_pipeline_rejects_invalid_graph_mutations_and_uncompiled_dispatch() -> None:
    ctx = MLContext()
    pipeline = MLPipeline(context=ctx)
    text = pipeline.input("text", dtype="string")
    output = _only_packet(
        pipeline.add(PassthroughNode(input_names=("text",)), inputs=[text])
    )

    with pytest.raises(PipelineNotCompiledError):
        pipeline.dispatch({"text": "hello"})

    pipeline.compile(outputs=[output])

    with pytest.raises(InvalidPipelineGraphError):
        pipeline.input("other")

    with pytest.raises(InvalidMLPacketError):
        pipeline.dispatch({"text": "hello", "extra": "world"})


def test_pipeline_rejects_foreign_packets_and_empty_outputs() -> None:
    ctx = MLContext()
    left_pipeline = MLPipeline(context=ctx)
    right_pipeline = MLPipeline(context=ctx)

    left = left_pipeline.input("left", dtype="string")
    foreign = right_pipeline.input("foreign", dtype="string")

    with pytest.raises(InvalidPipelineGraphError):
        left_pipeline.add(PassthroughNode(input_names=("left",)), inputs=[foreign])

    with pytest.raises(InvalidPipelineGraphError):
        left_pipeline.compile(outputs=[])

    with pytest.raises(InvalidPipelineGraphError):
        left_pipeline.input("left")

    assert left.name == "left"


def test_pipeline_compile_detects_unknown_dependencies_and_cycles() -> None:
    ctx = MLContext()

    invalid_pipeline = MLPipeline(context=ctx)
    source = invalid_pipeline.input("source", dtype="string")
    result = _only_packet(
        invalid_pipeline.add(
            PassthroughNode(input_names=("source",), output_names=("result",)),
            inputs=[source],
        )
    )
    invalid_pipeline._steps[0] = replace(  # pyright: ignore[reportPrivateUsage]
        invalid_pipeline._steps[0],  # pyright: ignore[reportPrivateUsage]
        input_names=("missing",),
    )

    with pytest.raises(InvalidPipelineGraphError):
        invalid_pipeline.compile(outputs=[result])

    cycle_pipeline = MLPipeline(context=ctx)
    cycle_input = cycle_pipeline.input("input", dtype="string")
    middle = _only_packet(
        cycle_pipeline.add(
            PassthroughNode(
                name="first",
                input_names=("input",),
                output_names=("middle",),
            ),
            inputs=[cycle_input],
        ),
    )
    final = _only_packet(
        cycle_pipeline.add(
            PassthroughNode(
                name="second",
                input_names=("middle",),
                output_names=("final",),
            ),
            inputs=[middle],
        ),
    )
    cycle_pipeline._steps[0] = replace(  # pyright: ignore[reportPrivateUsage]
        cycle_pipeline._steps[0],  # pyright: ignore[reportPrivateUsage]
        input_names=("final",),
    )

    with pytest.raises(InvalidPipelineGraphError):
        cycle_pipeline.compile(outputs=[final])


def test_node_execution_error_includes_node_name() -> None:
    ctx = MLContext()
    pipeline = MLPipeline(context=ctx)
    value = pipeline.input("value", dtype="string")
    output = _only_packet(pipeline.add(FailingNode(), inputs=[value]))
    pipeline.compile(outputs=[output])

    with pytest.raises(NodeExecutionError, match="failing"):
        pipeline.dispatch({"value": "hello"})
