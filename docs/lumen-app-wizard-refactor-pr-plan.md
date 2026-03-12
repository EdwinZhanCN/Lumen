# Lumen App 向导重构 PR 跟踪计划

> 目标：把当前向导流程中的系统性问题拆分为一组可审查、可回滚、可验证的 PR，按优先级逐步落地。
>
> 适用范围：`lumen-app`（FastAPI 控制平面 + `web-ui` 向导流程）

## 总体策略

- 先止血：优先修复“错误状态也能继续流程”“状态显示与真实安装路径不一致”等高风险问题。
- 再分层：把 API 路由中的编排逻辑下沉到 service/orchestrator，路由只负责协议层。
- 后扩展：在稳定流程上补可恢复安装、硬件能力明细、自动化测试与发布门禁。

## 里程碑状态

- [ ] M1: 流程正确性修复（PR-01 ~ PR-02）
- [ ] M2: 后端职责重构（PR-03 ~ PR-04）
- [x] M3: 硬件与预设体系重构（PR-05）
- [x] M4: 前端向导体验重构（PR-06）
- [ ] M5: 测试与 CI 门禁（PR-07）

---

## PR-01 会话路径与安装状态统一（止血）

**目标**
- 统一以会话路径 `cache_dir` 作为 install/config/server 的状态上下文，消除固定 `~/.lumen` 引发的误判。

**范围**
- `lumen-app/src/lumen_app/api/install.py`
- `lumen-app/web-ui/src/lib/api.ts`
- `lumen-app/web-ui/src/views/Install.tsx`
- `lumen-app/web-ui/src/views/SessionHub.tsx`（必要时）

**改动要点**
- [x] `GET /api/v1/install/status` 支持显式 `cache_dir` 参数（或通过统一会话上下文注入）。
- [x] `Install` 页面请求状态时传当前会话路径。
- [x] 后端环境检测逻辑不再硬编码 `~/.lumen`。
- [x] API 返回字段语义与 UI 展示一一对应（避免“显示未安装但实际已安装”）。

**验收标准**
- [ ] 同一机器上切换不同路径时，安装状态结果正确切换。
- [ ] 旧路径不再污染新会话路径状态。

**风险/回滚**
- 风险：接口变更影响前端兼容。
- 回滚点：保留旧参数兼容分支（短期）。

---

## PR-02 配置生成成功判定修复（止血）

**目标**
- 杜绝“配置生成失败但可进入下一步”。

**范围**
- `lumen-app/src/lumen_app/api/config.py`
- `lumen-app/web-ui/src/views/Config.tsx`
- `lumen-app/web-ui/src/context/WizardProvider.tsx`（必要时）
- `lumen-app/web-ui/src/lib/api.ts`

**改动要点**
- [x] 后端失败返回规范化（优先 HTTP 4xx/5xx + 错误体）。
- [x] 前端仅在 `success === true` 时置 `configGenerated=true`。
- [x] 修复同参数失败后无法重试的问题（`lastConfigKeyRef` 策略调整）。
- [x] 错误解析优先 `detail/message`，保证可见具体失败原因。

**验收标准**
- [ ] 人工构造无效 preset/无权限路径时，不能进入安装步骤。
- [ ] 同一参数可重复触发“重新生成配置”。

**风险/回滚**
- 风险：前后端错误协议兼容问题。
- 回滚点：保留旧响应字段兼容读取。

---

## PR-03 安装编排器抽离（职责分层）

**目标**
- 将 `api/install.py` 的流程编排职责下沉到 service 层，路由只做输入输出协议转换。

**范围**
- 新增：`lumen-app/src/lumen_app/services/install_orchestrator.py`
- 调整：`lumen-app/src/lumen_app/api/install.py`
- 调整：`lumen-app/src/lumen_app/services/state.py`

**改动要点**
- [x] 引入 `InstallOrchestrator`：负责 plan/run/retry/query/logs。
- [x] API 层保留参数校验、HTTP 映射、response model。
- [x] 安装任务存储抽象为独立仓储接口（内存版实现先行）。
- [ ] 清理旧任务模型与新任务模型并存问题（统一一套）。

**验收标准**
- [ ] `api/install.py` 明显瘦身，核心逻辑可单测。
- [ ] 安装流程不依赖路由内部私有函数。

**风险/回滚**
- 风险：拆分时序错误导致任务状态异常。
- 回滚点：保留原流程入口开关（临时 feature flag）。

---

## PR-04 安装状态机与可恢复机制（功能缺陷修复）

**目标**
- 让安装流程可恢复、可重试、可诊断，不再依赖步骤名称字符串匹配。

**范围**
- `lumen-app/src/lumen_app/services/install_orchestrator.py`（PR-03 后）
- `lumen-app/src/lumen_app/schemas/install.py`
- `lumen-app/src/lumen_app/api/install.py`
- `lumen-app/web-ui/src/views/Install.tsx`

**改动要点**
- [x] 步骤模型增加稳定 `step_id`。
- [x] 流程状态机标准化：`pending/running/completed/failed/cancelled`。
- [x] 前端提供“失败后重试当前任务/重建任务”入口。
- [x] 安装前置检查（配置文件存在、路径可写、环境可用）提前失败。

**验收标准**
- [ ] 缺配置文件时在安装开始前即明确失败。
- [ ] 失败可重试，无需刷新页面重开流程。

**风险/回滚**
- 风险：旧任务数据兼容。
- 回滚点：读取旧字段兼容映射。

---

## PR-05 预设系统重构与硬件检测修复（系统性）

**目标**
- 去除脆弱的反射式 preset 发现，修复带参 preset（如 Rockchip）缺失与驱动检测不稳定。

**范围**
- `lumen-app/src/lumen_app/utils/preset_registry.py`
- `lumen-app/src/lumen_app/services/config.py`
- `lumen-app/src/lumen_app/api/hardware.py`
- `lumen-app/web-ui/src/views/Hardware.tsx`

**改动要点**
- [x] 预设改为显式注册（支持参数化工厂与平台约束）。
- [x] `hardware/info` 返回每个 preset 的可用性与驱动明细。
- [x] 前端选择 preset 时显示驱动状态与安装建议。
- [x] 修复前端 preset 图标 key 与后端 preset 名称不匹配。

**验收标准**
- [ ] Rockchip 等带参预设可见且可选择。
- [ ] 推荐预设、驱动状态、可安装性显示一致。

**风险/回滚**
- 风险：已有 preset 名称兼容性。
- 回滚点：保留旧名称 alias 映射。

---

## PR-06 前端向导流程守卫与体验修复

**目标**
- 强化步骤约束和错误反馈，避免路由直跳、隐式回退、卡死状态。

**范围**
- `lumen-app/web-ui/src/context/WizardProvider.tsx`
- `lumen-app/web-ui/src/components/wizard/WizardLayout.tsx`
- `lumen-app/web-ui/src/views/Welcome.tsx`
- `lumen-app/web-ui/src/hooks/usePathCheck.ts`
- `lumen-app/web-ui/src/App.tsx`

**改动要点**
- [x] 增加基于前置步骤完成度的路由守卫。
- [x] Welcome 增加端口/服务名强校验，不再静默回退默认端口。
- [x] 修复 `usePathCheck` 中使用 `useState` 触发副作用的 bug，改为 `useEffect`。
- [x] 增强错误展示：网络错误、业务错误、权限错误分层提示。

**验收标准**
- [ ] 不满足前置条件无法进入后续步骤。
- [ ] 输入非法端口/服务名时有明确阻断与提示。

**风险/回滚**
- 风险：守卫过严影响已有体验。
- 回滚点：允许 debug 环境下关闭守卫。

---

## PR-07 测试与质量门禁（收口）

**目标**
- 建立可回归验证体系，防止后续改动重复引入流程问题。

**范围**
- 新增后端测试：`lumen-app/tests/...`
- 新增前端测试：`lumen-app/web-ui`（Vitest/Playwright 方案二选一或混合）
- CI 配置（仓库现有工作流）

**改动要点**
- [ ] API 合同测试：config/install/hardware/server 核心接口。
- [ ] Orchestrator 单测：安装状态机、失败重试、日志输出。
- [ ] 向导 E2E：`OpenPath -> Welcome -> Hardware -> Config -> Install` 主链路。
- [ ] 将 `ruff check`、`ty check`、pytest、前端 lint/typecheck 纳入阻断。

**验收标准**
- [ ] 覆盖关键分支：成功、失败、重试、路径切换。
- [ ] CI 全绿后方可合并。

**风险/回滚**
- 风险：测试基建首次引入成本。
- 回滚点：先核心回归场景，逐步扩大覆盖。

---

## PR 依赖关系

- PR-01、PR-02 可并行（建议先 PR-02）
- PR-03 依赖 PR-01/PR-02 完成
- PR-04 依赖 PR-03
- PR-05 可与 PR-03 后半程并行，但建议在 PR-04 前完成接口对齐
- PR-06 依赖 PR-01/PR-02，部分可并行
- PR-07 最后收口

---

## 执行跟踪清单（每个 PR 复用）

### 开发前
- [ ] 明确变更范围与非目标
- [ ] 明确兼容策略（旧字段/旧接口）

### 开发中
- [ ] 更新类型定义与 schema
- [ ] 同步前后端错误协议
- [ ] 补最小必要测试

### 提交前检查
- [ ] `ruff check`
- [ ] `ty check`
- [ ] `pytest`（至少变更相关测试）
- [ ] 前端 `lint + typecheck + build`

### 合并后验证
- [ ] 手工走通主流程
- [ ] 回归“路径切换 + 失败重试 + 重新配置”
- [ ] 更新文档状态

---

## 当前状态记录

- 文档版本：`v1`
- 更新时间：`2026-03-06`
- 当前阶段：`PR-06 代码改动完成，待验收；下一步 PR-07`
