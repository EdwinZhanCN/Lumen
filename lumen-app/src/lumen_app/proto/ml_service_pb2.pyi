from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CODE_UNSPECIFIED: _ClassVar[ErrorCode]
    ERROR_CODE_INVALID_ARGUMENT: _ClassVar[ErrorCode]
    ERROR_CODE_UNAVAILABLE: _ClassVar[ErrorCode]
    ERROR_CODE_DEADLINE_EXCEEDED: _ClassVar[ErrorCode]
    ERROR_CODE_INTERNAL: _ClassVar[ErrorCode]

ERROR_CODE_UNSPECIFIED: ErrorCode
ERROR_CODE_INVALID_ARGUMENT: ErrorCode
ERROR_CODE_UNAVAILABLE: ErrorCode
ERROR_CODE_DEADLINE_EXCEEDED: ErrorCode
ERROR_CODE_INTERNAL: ErrorCode

class Error(_message.Message):
    __slots__ = ("code", "message", "detail")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    message: str
    detail: str
    def __init__(
        self,
        code: ErrorCode | str | None = ...,
        message: str | None = ...,
        detail: str | None = ...,
    ) -> None: ...

class IOTask(_message.Message):
    __slots__ = ("name", "input_mimes", "output_mimes", "limits")
    class LimitsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

    NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_MIMES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MIMES_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    name: str
    input_mimes: _containers.RepeatedScalarFieldContainer[str]
    output_mimes: _containers.RepeatedScalarFieldContainer[str]
    limits: _containers.ScalarMap[str, str]
    def __init__(
        self,
        name: str | None = ...,
        input_mimes: _Iterable[str] | None = ...,
        output_mimes: _Iterable[str] | None = ...,
        limits: _Mapping[str, str] | None = ...,
    ) -> None: ...

class Capability(_message.Message):
    __slots__ = (
        "service_name",
        "model_ids",
        "runtime",
        "max_concurrency",
        "precisions",
        "extra",
        "tasks",
        "protocol_version",
    )
    class ExtraEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_IDS_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENCY_FIELD_NUMBER: _ClassVar[int]
    PRECISIONS_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    model_ids: _containers.RepeatedScalarFieldContainer[str]
    runtime: str
    max_concurrency: int
    precisions: _containers.RepeatedScalarFieldContainer[str]
    extra: _containers.ScalarMap[str, str]
    tasks: _containers.RepeatedCompositeFieldContainer[IOTask]
    protocol_version: str
    def __init__(
        self,
        service_name: str | None = ...,
        model_ids: _Iterable[str] | None = ...,
        runtime: str | None = ...,
        max_concurrency: int | None = ...,
        precisions: _Iterable[str] | None = ...,
        extra: _Mapping[str, str] | None = ...,
        tasks: _Iterable[IOTask | _Mapping] | None = ...,
        protocol_version: str | None = ...,
    ) -> None: ...

class InferRequest(_message.Message):
    __slots__ = (
        "correlation_id",
        "task",
        "payload",
        "meta",
        "payload_mime",
        "seq",
        "total",
        "offset",
    )
    class MetaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_MIME_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    correlation_id: str
    task: str
    payload: bytes
    meta: _containers.ScalarMap[str, str]
    payload_mime: str
    seq: int
    total: int
    offset: int
    def __init__(
        self,
        correlation_id: str | None = ...,
        task: str | None = ...,
        payload: bytes | None = ...,
        meta: _Mapping[str, str] | None = ...,
        payload_mime: str | None = ...,
        seq: int | None = ...,
        total: int | None = ...,
        offset: int | None = ...,
    ) -> None: ...

class InferResponse(_message.Message):
    __slots__ = (
        "correlation_id",
        "is_final",
        "result",
        "meta",
        "error",
        "seq",
        "total",
        "offset",
        "result_mime",
        "result_schema",
    )
    class MetaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: str | None = ..., value: str | None = ...) -> None: ...

    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    RESULT_MIME_FIELD_NUMBER: _ClassVar[int]
    RESULT_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    correlation_id: str
    is_final: bool
    result: bytes
    meta: _containers.ScalarMap[str, str]
    error: Error
    seq: int
    total: int
    offset: int
    result_mime: str
    result_schema: str
    def __init__(
        self,
        correlation_id: str | None = ...,
        is_final: bool = ...,
        result: bytes | None = ...,
        meta: _Mapping[str, str] | None = ...,
        error: Error | _Mapping | None = ...,
        seq: int | None = ...,
        total: int | None = ...,
        offset: int | None = ...,
        result_mime: str | None = ...,
        result_schema: str | None = ...,
    ) -> None: ...
