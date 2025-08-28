import enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import abc

class RuntimeKind(str, enum.Enum):
    ONNXRT = "onnxrt"
    RKNN = "rknn"
    ARMNN = "armnn"

@dataclass
class BackendInfo:
    runtime: str
    device: Optional[str] = None
    pack_name: Optional[str] = None
    face_embedding_dim: Optional[int] = None
    precisions: List[str] = field(default_factory=list)
    max_batch_size: Optional[int] = None  # backend hint (if any)
    supports_image_batch: bool = False
    extra: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str,object]:
        return {
            'runtime': self.runtime,
            'device': self.device,
            'pack_name': self.pack_name,
            'face_embedding_dim': self.face_embedding_dim,
            'precisions': self.precisions,
            'max_batch_size': self.max_batch_size,
            'supports_image_batch': self.supports_image_batch,
            'extra': self.extra
        }

class BaseFaceBackend(abc.ABC):
    def __init__(
        self,
        pack_name: Optional[str] = None,
        device_preference: Optional[str] = None,
        max_batch_size: Optional[int] = None,
    ) -> None:
        self._initialize: bool = False
