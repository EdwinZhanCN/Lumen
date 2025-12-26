"""
Enhanced Runner View - Main application state management with button management

集中管理所有按钮的状态和业务逻辑，提供数据绑定机制
"""

# import os is not used
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import flet as ft
import yaml

from ...utils.logger import get_logger
from ..components.button_container import ButtonContainer
from ..components.continue_button import ContinueButton
from ..components.reset_button import ResetButton
from ..i18n_manager import t

logger = get_logger("lumen.ui.runner_view")


class RunnerState(Enum):
    """Application states"""

    WELCOME = "welcome"
    DEVICE_CONF = "device_conf"
    PRESETS = "presets"
    ADVANCED = "advanced"
    INSTALLER = "installer"
    ACTIVE = "active"


class WorkflowType(Enum):
    """Workflow types"""

    LUMILIO_PHOTOS = "lumilio_photos"
    ADVANCED = "advanced"


class ViewDataBinding:
    """视图数据绑定辅助类"""

    def __init__(self, view_name: str, runner_view: "RunnerView"):
        """
        初始化数据绑定

        Args:
            view_name: 视图名称
            runner_view: RunnerView 实例
        """
        self.view_name = view_name
        self.runner_view = runner_view
        self.data = {}
        self.listeners = []

    def set_data(self, key: str, value: Any):
        """
        设置数据并通知监听器

        Args:
            key: 数据键
            value: 数据值
        """
        self.data[key] = value
        # 更新 runner_view 中的数据
        self.runner_view.view_data[self.view_name] = self.data
        self._notify_listeners(key, value)

    def get_data(self, key: str, default=None):
        """
        获取数据

        Args:
            key: 数据键
            default: 默认值

        Returns:
            数据值
        """
        return self.data.get(key, default)

    def add_listener(self, callback: Callable[[str, Any], None]):
        """
        添加数据变化监听器

        Args:
            callback: 监听器回调函数，接收 (key, value) 参数
        """
        self.listeners.append(callback)

    def _notify_listeners(self, key: str, value: Any):
        """通知所有监听器"""
        for listener in self.listeners:
            try:
                listener(key, value)
            except Exception as e:
                logger.error(f"Listener error: {e}")


class ButtonManager:
    """集中管理所有按钮的状态和回调"""

    def __init__(self, runner_view: "RunnerView"):
        """
        初始化按钮管理器

        Args:
            runner_view: RunnerView 实例
        """
        self.runner_view = runner_view
        self.buttons: Dict[str, Dict[str, Any]] = {}  # 存储各视图的按钮引用
        self.containers: Dict[str, ButtonContainer] = {}  # 存储按钮容器

    def register_view_buttons(
        self, view_name: str, continue_btn: ContinueButton, reset_btn: ResetButton
    ) -> ButtonContainer:
        """
        注册视图的按钮并创建容器

        Args:
            view_name: 视图名称
            continue_btn: 继续按钮
            reset_btn: 重置按钮

        Returns:
            ButtonContainer: 按钮容器
        """
        # 存储按钮引用
        self.buttons[view_name] = {"continue": continue_btn, "reset": reset_btn}

        # 创建容器
        container = ButtonContainer(continue_btn, reset_btn)
        self.containers[view_name] = container

        # 绑定回调
        continue_btn.get_button().on_click = self._create_continue_handler(view_name)
        reset_btn.get_button().on_click = self._create_reset_handler(view_name)

        logger.debug(f"Registered buttons for view: {view_name}")
        return container

    def _create_continue_handler(self, view_name: str):
        """
        创建继续按钮的处理函数

        Args:
            view_name: 视图名称

        Returns:
            处理函数
        """

        def handler(e):
            self.runner_view.handle_continue_action(view_name)

        return handler

    def _create_reset_handler(self, view_name: str):
        """
        创建重置按钮的处理函数

        Args:
            view_name: 视图名称

        Returns:
            处理函数
        """

        def handler(e):
            self.runner_view.handle_reset_action(view_name)

        return handler

    def update_button_state(
        self,
        view_name: str,
        continue_disabled: Optional[bool] = None,
        reset_disabled: Optional[bool] = None,
        continue_text: Optional[str] = None,
        reset_text: Optional[str] = None,
    ):
        """
        更新按钮状态

        Args:
            view_name: 视图名称
            continue_disabled: 继续按钮是否禁用
            reset_disabled: 重置按钮是否禁用
            continue_text: 继续按钮文本
            reset_text: 重置按钮文本
        """
        if view_name in self.buttons:
            if continue_disabled is not None:
                self.buttons[view_name]["continue"].update_disabled(continue_disabled)
            if reset_disabled is not None:
                self.buttons[view_name]["reset"].update_disabled(reset_disabled)
            if continue_text is not None:
                self.buttons[view_name]["continue"].update_text(continue_text)
            if reset_text is not None:
                self.buttons[view_name]["reset"].update_text(reset_text)

            # 更新容器
            if view_name in self.containers:
                self.containers[view_name].update_container()

    def get_container(self, view_name: str) -> Optional[ButtonContainer]:
        """
        获取指定视图的按钮容器

        Args:
            view_name: 视图名称

        Returns:
            按钮容器或 None
        """
        return self.containers.get(view_name)


class RunnerStateMachine:
    """状态机，管理应用程序流程"""

    def __init__(self, cache_directory: str = "~/.lumen"):
        self.cache_directory = Path(cache_directory).expanduser()
        self.current_state = RunnerState.WELCOME
        self.workflow_type: Optional[WorkflowType] = None
        self.state_changed_callbacks: list[Callable] = []

        # 确定初始状态
        self._determine_initial_state()

    def _determine_initial_state(self):
        """检查现有配置以确定初始状态"""
        config_path = self.cache_directory / "lumen-config.yaml"

        if not config_path.exists():
            self.current_state = RunnerState.WELCOME
        elif self._check_modules_installed():
            self.current_state = RunnerState.ACTIVE
        else:
            self.current_state = RunnerState.INSTALLER

    def _check_modules_installed(self) -> bool:
        """检查是否已安装所需模块"""
        models_dir = self.cache_directory / "models"
        return models_dir.exists() and len(list(models_dir.iterdir())) > 0

    def add_state_changed_callback(self, callback: Callable):
        """添加状态变化回调"""
        self.state_changed_callbacks.append(callback)

    def notify_state_changed(self):
        """通知所有回调状态已变化"""
        for callback in self.state_changed_callbacks:
            callback(self.current_state)

    def transition_to_state(self, state: RunnerState):
        """转换到新状态"""
        if self.current_state != state:
            logger.debug(
                f"State transition: {self.current_state.value} -> {state.value}"
            )
            self.current_state = state
            self.notify_state_changed()

    # 状态转换处理器
    def on_welcome_lumilio_photos(self, cache_dir: Optional[str] = None):
        """处理 Lumilio Photos 工作流选择"""
        if cache_dir:
            self.cache_directory = Path(cache_dir).expanduser()
        self.workflow_type = WorkflowType.LUMILIO_PHOTOS
        self.transition_to_state(RunnerState.DEVICE_CONF)

    def on_welcome_advanced_mode(self, cache_dir: Optional[str] = None):
        """处理高级模式工作流选择"""
        if cache_dir:
            self.cache_directory = Path(cache_dir).expanduser()
        self.workflow_type = WorkflowType.ADVANCED
        self.transition_to_state(RunnerState.ADVANCED)

    def on_device_conf_completed(self):
        """处理设备配置完成"""
        self.transition_to_state(RunnerState.PRESETS)

    def on_presets_completed(self):
        """处理预设选择完成"""
        self.transition_to_state(RunnerState.INSTALLER)

    def on_advanced_completed(self):
        """处理高级配置完成"""
        self.transition_to_state(RunnerState.INSTALLER)

    def on_installer_completed(self):
        """处理安装完成"""
        self.transition_to_state(RunnerState.ACTIVE)

    def on_reset_to_welcome(self):
        """重置到欢迎状态"""
        self.workflow_type = None
        self.transition_to_state(RunnerState.WELCOME)


class RunnerView:
    """增强的 RunnerView，集中管理所有逻辑"""

    def __init__(self, cache_directory: str = "~/.lumen"):
        """
        初始化 RunnerView

        Args:
            cache_directory: 缓存目录
        """
        self.cache_directory = cache_directory
        self.state_machine = RunnerStateMachine(cache_directory)
        self.main_container: ft.Container = ft.Container()

        # 新增组件
        self.button_manager = ButtonManager(self)
        self.view_data: Dict[str, Dict[str, Any]] = {}  # 存储各视图的数据
        self.view_bindings: Dict[str, ViewDataBinding] = {}  # 存储数据绑定

        # 设置状态变化处理
        self.state_machine.add_state_changed_callback(self.update_view)

    def handle_continue_action(self, view_name: str):
        """
        处理继续按钮点击

        Args:
            view_name: 视图名称
        """
        logger.debug(f"Continue action from view: {view_name}")

        # 根据视图类型处理不同的业务逻辑
        if view_name == "device_conf":
            self._handle_device_conf_continue()
        elif view_name == "presets":
            self._handle_presets_continue()
        elif view_name == "advanced":
            self._handle_advanced_continue()
        else:
            logger.warning(f"Unknown view: {view_name}")

    def handle_reset_action(self, view_name: str):
        """
        处理重置按钮点击

        Args:
            view_name: 视图名称
        """
        logger.debug(f"Reset action from view: {view_name}")

        # 重置当前视图的数据
        if view_name in self.view_data:
            del self.view_data[view_name]
        if view_name in self.view_bindings:
            self.view_bindings[view_name].data.clear()

        # 返回欢迎页
        self.state_machine.on_reset_to_welcome()

    def _handle_device_conf_continue(self):
        """处理设备配置继续"""
        # 获取设备配置数据
        device_config = self.view_data.get("device_conf", {})
        if device_config:
            logger.info(f"Device config saved: {device_config}")
            self.state_machine.on_device_conf_completed()
        else:
            logger.warning("No device configuration selected")

    def _handle_presets_continue(self):
        """处理预设选择继续"""
        # 获取预设数据
        preset_data = self.view_data.get("presets", {})
        if preset_data.get("selected_preset"):
            logger.info(f"Preset selected: {preset_data['selected_preset']}")
            self.state_machine.on_presets_completed()
        else:
            logger.warning("No preset selected")

    def _handle_advanced_continue(self):
        """处理高级配置继续"""
        # 获取高级配置数据
        advanced_config = self.view_data.get("advanced", {})
        if advanced_config:
            logger.info(f"Advanced config saved: {advanced_config}")
            self.state_machine.on_advanced_completed()
        else:
            logger.warning("No advanced configuration provided")

    def _handle_installer_continue(self):
        """处理安装器继续"""
        # 检查安装是否完成
        installer_data = self.view_data.get("installer", {})
        if installer_data.get("installation_complete"):
            # 转移到活动状态
            self.state_machine.on_installer_completed()
        else:
            logger.warning("Installation not complete")

    def create_data_binding(self, view_name: str) -> ViewDataBinding:
        """
        为视图创建数据绑定

        Args:
            view_name: 视图名称

        Returns:
            ViewDataBinding: 数据绑定实例
        """
        binding = ViewDataBinding(view_name, self)
        self.view_bindings[view_name] = binding
        return binding

    def create_view(self) -> ft.Container:
        """创建并返回主视图容器"""
        # 使用当前状态初始化（不更新，因为尚未添加到页面）
        self.update_view(self.state_machine.current_state, force_update=False)
        return self.main_container

    def update_view(self, state: RunnerState, force_update: bool = True):
        """
        基于当前状态更新视图

        Args:
            state: 当前状态
            force_update: 是否强制更新视图（首次创建时应为 False）
        """
        logger.debug(f"Updating view for state: {state.value}")

        # 清除旧的视图内容
        self.main_container.content = None

        # 根据状态创建新视图
        if state == RunnerState.WELCOME:
            self._create_welcome_view()
        elif state == RunnerState.DEVICE_CONF:
            self._create_device_conf_view()
        elif state == RunnerState.PRESETS:
            self._create_presets_view()
        elif state == RunnerState.ADVANCED:
            self._create_advanced_view()
        elif state == RunnerState.INSTALLER:
            self._create_installer_view()
        elif state == RunnerState.ACTIVE:
            self._create_active_view()

        # 强制更新视图（如果需要）
        if force_update:
            self._force_view_update()

    def _force_view_update(self):
        """强制更新视图"""
        try:
            if hasattr(self.main_container, "update"):
                self.main_container.update()
            elif hasattr(self.main_container, "page") and self.main_container.page:
                self.main_container.page.update()
            logger.debug("View updated successfully")
        except Exception as e:
            logger.error(f"Could not update view: {e}")

    def _create_welcome_view(self):
        """创建欢迎视图"""
        from .welcome_view import WelcomeView

        welcome_view = WelcomeView(
            lumilio_handler=lambda e,
            cache_dir: self.state_machine.on_welcome_lumilio_photos(cache_dir),
            advanced_handler=lambda e: self.state_machine.on_welcome_advanced_mode(),
        )

        self.main_container.content = welcome_view

    def _create_device_conf_view(self):
        """创建设备配置视图"""
        logger.debug("Creating device configuration view...")

        # 创建按钮组件
        continue_btn = ContinueButton()
        reset_btn = ResetButton()

        # 注册按钮到管理器
        button_container = self.button_manager.register_view_buttons(
            "device_conf", continue_btn, reset_btn
        )

        # 创建数据绑定
        data_binding = self.create_data_binding("device_conf")

        # 创建设备配置视图（传入按钮容器和数据绑定）
        from .device_conf_view import DeviceConfView

        device_conf_view = DeviceConfView(
            button_container=button_container,
            data_binding=data_binding,
            on_reset=lambda: self.state_machine.on_reset_to_welcome(),
        )

        self.main_container.content = device_conf_view
        logger.debug("Device configuration view set up complete")

    def _create_presets_view(self):
        """创建预设选择视图"""
        # 创建按钮组件
        continue_btn = ContinueButton()
        reset_btn = ResetButton()

        # 注册按钮到管理器
        button_container = self.button_manager.register_view_buttons(
            "presets", continue_btn, reset_btn
        )

        # 创建数据绑定
        data_binding = self.create_data_binding("presets")

        # 获取前序视图的数据
        device_config = self.view_data.get("device_conf", {}).get("config")
        region = self.view_data.get("welcome", {}).get("region", "cn")
        cache_dir = str(self.state_machine.cache_directory)

        if not device_config:
            logger.warning("No device configuration found")
            return

        # 创建预设视图
        from .presets_view import PresetsView

        presets_view = PresetsView(
            cache_dir=cache_dir,
            region=region,
            device_config=device_config,
            button_container=button_container,
            data_binding=data_binding,
            on_reset=lambda: self.state_machine.on_reset_to_welcome(),
        )

        self.main_container.content = presets_view

    def _create_advanced_view(self):
        """创建高级配置视图"""
        # 创建按钮组件
        continue_btn = ContinueButton()
        reset_btn = ResetButton()

        # 注册按钮到管理器
        button_container = self.button_manager.register_view_buttons(
            "advanced", continue_btn, reset_btn
        )

        # 创建数据绑定
        data_binding = self.create_data_binding("advanced")

        # 创建高级视图
        from .advanced_view import AdvancedView

        advanced_view = AdvancedView(
            button_container=button_container, data_binding=data_binding
        )

        self.main_container.content = advanced_view

    def _create_installer_view(self):
        """创建安装器视图"""
        # 获取前序视图的数据
        device_config = None
        lumen_config = None

        # 根据 workflow 类型获取配置
        if self.state_machine.workflow_type == WorkflowType.LUMILIO_PHOTOS:
            device_config = self.view_data.get("device_conf", {}).get("config")
            lumen_config = self.view_data.get("presets", {}).get("lumen_config")
        elif self.state_machine.workflow_type == WorkflowType.ADVANCED:
            device_config = self.view_data.get("advanced", {}).get("device_config")
            # TODO: 从 advanced config_dict 构建 lumen_config
            logger.warning("Advanced workflow config building not yet implemented")

        if not device_config:
            logger.warning("No device configuration found")
            return

        # 创建按钮组件
        continue_btn = ContinueButton(t("installer.complete"))
        reset_btn = ResetButton(t("installer.back_button"))

        button_container = self.button_manager.register_view_buttons(
            view_name="installer", continue_btn=continue_btn, reset_btn=reset_btn
        )

        # 绑定自定义处理程序
        continue_btn.get_button().on_click = lambda e: self._handle_installer_continue()
        reset_btn.get_button().on_click = (
            lambda e: self.state_machine.on_reset_to_welcome()
        )

        # 创建数据绑定
        data_binding = self.create_data_binding("installer")

        # 创建安装器视图（传入完整参数）
        from .installer_view import InstallerView

        installer_view = InstallerView(
            cache_dir=str(self.state_machine.cache_directory),
            device_config=device_config,
            lumen_config=lumen_config,
            button_container=button_container,
            data_binding=data_binding,
        )

        self.main_container.content = installer_view

    def _create_active_view(self):
        """创建活动/运行状态视图"""
        # 创建按钮组件
        continue_btn = ContinueButton(t("active_runner.view_logs"))
        reset_btn = ResetButton(t("active_runner.stop_server"))

        button_container = self.button_manager.register_view_buttons(
            view_name="active", continue_btn=continue_btn, reset_btn=reset_btn
        )

        # 绑定自定义处理程序
        reset_btn.get_button().on_click = (
            lambda e: self.state_machine.on_reset_to_welcome()
        )

        # 创建数据绑定
        data_binding = self.create_data_binding("active")

        # 获取配置
        lumen_config = self.view_data.get("installer", {}).get("lumen_config")

        # 如果配置不在内存中，从文件加载
        if not lumen_config:
            from lumen_resources import LumenConfig

            config_path = self.state_machine.cache_directory / "lumen-config.yaml"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config_data = yaml.safe_load(f)
                    lumen_config = LumenConfig(**config_data)
                    # 保存到 view_data 以便后续使用
                    self.view_data.setdefault("installer", {})["lumen_config"] = (
                        lumen_config
                    )
                except Exception as e:
                    logger.error(f"Failed to load config: {e}")
                    import traceback

                    traceback.print_exc()
                    return
            else:
                logger.warning("No LumenConfig found and no config file exists")
                return

        # 导入并使用活动运行器视图
        from .active_runner_view import ActiveRunnerView

        # 创建活动视图（传入完整参数）
        active_view = ActiveRunnerView(
            cache_dir=str(self.state_machine.cache_directory),
            lumen_config=lumen_config,
            button_container=button_container,
            data_binding=data_binding,
        )

        self.main_container.content = active_view

    def _save_config_file(self):
        """保存配置到文件"""
        config_path = self.state_machine.cache_directory / "lumen-config.yaml"

        if not config_path.exists():
            config_data = {
                "version": "1.0.0",
                "cache_dir": str(self.state_machine.cache_directory),
                "workflow_type": self.state_machine.workflow_type.value
                if self.state_machine.workflow_type
                else None,
                "created_at": str(Path(__file__).stat().st_mtime),
            }

            # 添加各视图的数据
            config_data["view_data"] = self.view_data

            config_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(
                        config_data, f, default_flow_style=False, allow_unicode=True
                    )
                logger.info(f"Configuration saved to: {config_path}")
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")


def create_runner_view(cache_directory: str = "~/.lumen"):
    """
    创建带状态管理的运行器视图的工厂函数

    Args:
        cache_directory: 用于缓存配置和模型的目录

    Returns:
        ft.Container: 带状态管理的主容器
    """
    runner = RunnerView(cache_directory)
    return runner.create_view()


def create_active_runner_view(cache_directory: str = "~/.lumen"):
    """
    兼容性函数 - 创建带活动状态检查的运行器视图

    Args:
        cache_directory: 用于缓存配置和模型的目录

    Returns:
        tuple: (main_container, state_machine) 用于向后兼容
    """
    runner = RunnerView(cache_directory)
    return runner.create_view(), runner.state_machine
