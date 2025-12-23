import flet as ft

from .i18n_manager import get_i18n_manager, t
from .view.monitor_view import create_monitor_view
from .view.runner_view import create_runner_view  # 内部将负责管理引导流


def main(page: ft.Page):
    # ---- 页面基础设置 ----
    page.title = t("app.title")
    page.theme_mode = ft.ThemeMode.SYSTEM

    try:
        page.window.min_width = 800
        page.window.min_height = 600
    except AttributeError:
        pass

    # 初始化 i18n
    i18n_manager = get_i18n_manager()

    # ---- 核心视图组件缓存 ----
    # 注意：现在 create_runner_view() 应该返回一个能管理内部状态（Welcome -> Presets -> Installer -> Active）的容器
    runner_flow_container = create_runner_view()
    monitor_dashboard = create_monitor_view()

    # ---- 内容区域容器 ----
    content_area = ft.Container(
        content=runner_flow_container,
        expand=True,
        padding=20,
    )

    # ---- 导航处理逻辑 ----
    def handle_navigation_change(e):
        index = e.control.selected_index
        if index == 0:
            content_area.content = runner_flow_container
        elif index == 1:
            content_area.content = monitor_dashboard
        page.update()

    # ---- 顶级导航栏 (Navigation Rail) ----
    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=200,
        destinations=[
            # 职责 1: 管理与启动本地服务 (包含所有引导步骤)
            ft.NavigationRailDestination(
                icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
                selected_icon=ft.Icons.PLAY_CIRCLE_FILLED,
                label=t("views.runner"),
            ),
            # 职责 2: 监控全局网格状态
            ft.NavigationRailDestination(
                icon=ft.Icons.MONITOR_HEART_OUTLINED,
                selected_icon=ft.Icons.MONITOR_HEART,
                label=t("views.monitor"),
            ),
        ],
        on_change=handle_navigation_change,
    )

    # ---- 语言切换逻辑 ----
    def change_language(locale):
        i18n_manager.set_locale(locale)
        page.title = t("app.title")

        # 仅更新两个顶级标签的文本
        if rail.destinations:
            rail.destinations[0].label = t("views.runner")
            rail.destinations[1].label = t("views.monitor")

        # # 触发当前视图的内部翻译更新 (如果视图支持 refresh 方法)
        # if hasattr(content_area.content, "refresh_i18n"):
        #     content_area.content.refresh_i18n()

        page.update()

    lang_switch = ft.PopupMenuButton(
        icon=ft.Icons.LANGUAGE,
        tooltip="Change Language",
        items=[
            ft.PopupMenuItem(text="English", on_click=lambda _: change_language("en")),
            ft.PopupMenuItem(text="中文", on_click=lambda _: change_language("zh")),
        ],
    )

    # ---- 主布局组装 ----
    page.add(
        ft.Row(
            [
                rail,
                ft.VerticalDivider(width=1),
                ft.Column(
                    [
                        # 顶部工具栏：仅包含语言切换
                        ft.Container(
                            content=lang_switch,
                            alignment=ft.alignment.top_right,
                            padding=ft.padding.only(right=10, top=10),
                        ),
                        # 主内容区
                        content_area,
                    ],
                    expand=True,
                ),
            ],
            expand=True,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
