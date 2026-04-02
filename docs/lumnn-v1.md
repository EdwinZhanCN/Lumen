# LumNN v1

LumNN is a high-level orchestration API for Lumen. It is inspired by graph-based
ML runtimes such as WebNN, but it does not aim to be a low-level operator
runtime, compiler, or universal memory manager.

## Design Goals

- Provide a strong code-level contract for pipeline orchestration.
- Keep the public surface intentionally small: `MLPacket`, `MLNode`,
  `MLContext`, and `MLPipeline`.
- Make future package migrations incremental by preserving existing backend and
  business logic while replacing weak glue abstractions.
- Support device-aware execution with best-effort packet transfer semantics.

## Non-Goals

- No cross-process zero-copy guarantees.
- No operator fusion, graph optimization, or low-level kernel scheduling.
- No replacement for `lumen-resources` model download and configuration flows.

## Minimal Example

```python
from lumen.nn import MLContext, MLPipeline
from lumen.nn._testing import JoinTextNode, UppercaseTextNode

ctx = MLContext(preferred_device="auto")
pipeline = MLPipeline(context=ctx)

left = pipeline.input("left", dtype="string")
right = pipeline.input("right", dtype="string")

upper_left = pipeline.add(UppercaseTextNode(), inputs=[left])
joined = pipeline.add(
    JoinTextNode(separator=" / ", input_names=("upper_text", "right")),
    inputs=[upper_left, right],
)

pipeline.compile(outputs=[joined])
result = pipeline.dispatch({"left": "lumen", "right": "graph"})

assert result["joined"].value == "LUMEN / graph"
```

## Execution Model

- `MLPipeline.input()` creates symbolic input packets.
- `MLPipeline.add()` wires `MLNode` instances into a DAG and returns symbolic
  output packets.
- `MLPipeline.compile()` validates the graph and freezes it.
- `MLPipeline.dispatch()` materializes runtime packets, initializes nodes,
  executes the graph in topological order, and returns declared outputs.

## Device Semantics

- `MLContext.resolve_device()` normalizes requested devices and falls back to
  CPU when a supported non-CPU device is unavailable.
- `MLContext.transfer()` is best effort in v1. It returns a new packet with an
  updated logical device but does not promise physical zero-copy behavior.
