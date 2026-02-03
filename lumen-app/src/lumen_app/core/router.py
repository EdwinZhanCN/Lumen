# router.py
import grpc

from lumen_app.proto import ml_service_pb2, ml_service_pb2_grpc
from lumen_app.utils.logger import get_logger

logger = get_logger("lumen.router")


class HubRouter(ml_service_pb2_grpc.InferenceServicer):
    def __init__(self, services: list):
        self.services = services
        # 建立 Task Key -> Service 实例的映射
        self._route_table = {}
        for svc in services:
            for task_key in svc.get_supported_tasks():
                # 如果 key 已存在，这里可以选择附加到列表或简单的覆盖
                # 既然你说交给 SDK 判断，Hub 这里默认选择第一个匹配的服务
                if task_key not in self._route_table:
                    self._route_table[task_key] = svc

    async def Infer(self, request_iterator, context):
        """多路复用分发推理请求"""
        try:
            # 获取流的第一条消息以识别 Task
            first_req = await request_iterator.__anext__()
        except StopAsyncIteration:
            return

        task_key = first_req.task

        target_svc = self._route_table.get(task_key)

        if not target_svc:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Task {task_key} not supported")

        if target_svc is not None:
            # 构造包装后的迭代器透传给子服务
            async def stream_wrapper():
                yield first_req
                async for req in request_iterator:
                    yield req

            # 零拷贝转发流式响应
            async for resp in target_svc.Infer(stream_wrapper(), context):
                yield resp

    async def GetCapabilities(self, request, context):
        """汇总所有子服务的能力宣告"""
        all_tasks = []
        for svc in self.services:
            caps = await svc.GetCapabilities(request, context)
            all_tasks.extend(caps.tasks)
        return ml_service_pb2.Capability(tasks=all_tasks)

    def attach_to_server(self, server: grpc.Server):
        """
        Attach the hub router to the gRPC server.

        This registers the router as the single InferenceServicer that handles
        all incoming requests and routes them to appropriate services.

        Args:
            server: The gRPC server instance to attach to
        """
        ml_service_pb2_grpc.add_InferenceServicer_to_server(self, server)
        logger.info(
            f"HubRouter attached to server with {len(self.services)} service(s)"
        )
        logger.debug(f"Route table: {list(self._route_table.keys())}")
