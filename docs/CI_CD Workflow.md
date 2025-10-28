# Lumen 开发指南

## 项目结构

```
Lumen/
├── lumen-clip/          # CLIP 模型服务
├── lumen-face/          # 人脸识别服务
├── lumen-ocr/           # OCR 文字识别服务
├── lumen-resources/     # 统一资源管理
└── .github/workflows/   # CI/CD 配置
```

## 分支策略

### 分支说明

- **`main`**：稳定分支，代表可发布的版本
- **`dev`**：开发分支，用于日常开发和集成测试
- **feature branches**：功能分支，从 `dev` 分出，合并回 `dev`

### 工作流程

#### 1. 日常开发
```bash
# 切换到 dev 分支
git checkout dev
# 从 main 合并 （ tag 在 main 中更新，所以必须合并)
git pull origin dev

# 创建功能分支（可选）
git checkout -b feature/new-feature
# 开发完成后合并回 dev
git checkout dev
git merge feature/new-feature
git push origin dev
```

#### 2. 自动发布（Dev Prerelease）
每次 push 到 `dev` 分支会自动触发：
- 构建所有子包
- 创建 prerelease（如 `v0.1.2.dev5+g1234567-dev`）
- 发布到 GitHub Releases

#### 3. 正式发布
```bash
# 1. 合并 dev 到 main
git checkout main
git merge dev
git push origin main
# → 触发 CI 构建（验证，不发布）

# 2. 创建正式 tag
git tag v1.0.0
git push origin v1.0.0
# → 自动构建并发布正式 release
```

## 版本管理

### 版本号格式

- **开发版本**：`v0.1.2.dev5+g1234567-dev`
  - `0.1.2`：基础版本号
  - `dev5`：第 5 个开发提交
  - `+g1234567`：Git commit hash
  - `-dev`：开发版本标识

- **正式版本**：`v1.0.0`
  - 直接使用 Git tag 名称

### 自动化规则

1. **dev 分支**：每次 push 自动生成开发版本号
2. **tag 推送**：使用 tag 名称作为版本号
3. **版本一致性**：所有子包使用相同版本号

## CI/CD 工作流

### 1. CI Build (`.github/workflows/ci.yml`)
- **触发条件**：push 到 main、PR 到 main/dev
- **功能**：构建验证、测试导入
- **结果**：不创建 release，仅验证代码质量

### 2. Release Packages (`.github/workflows/release.yml`)
- **触发条件**：push 到 dev、推送 v* tag
- **功能**：构建所有包、创建 release
- **结果**：
  - dev 分支 → Prerelease
  - tag 推送 → 正式 Release

## 构建说明

### 本地构建
```bash
# 构建单个包
cd lumen-clip
python -m build --outdir ../dist

# 构建所有包
for pkg in lumen-clip lumen-resources lumen-face lumen-ocr; do
  cd $pkg && python -m build --outdir ../dist && cd ..
done
```

### 依赖管理

所有子包使用统一的构建依赖：
- `setuptools>=45`
- `setuptools_scm[toml]>=6.2`
- `build`

## 发布清单

正式发布前的检查清单：

1. **代码质量**：
   - [ ] 所有测试通过
   - [ ] CI 构建成功
   - [ ] 代码审查完成

2. **版本准备**：
   - [ ] 更新 CHANGELOG.md
   - [ ] 确认版本号
   - [ ] 合并 dev 到 main

3. **发布步骤**：
   - [ ] 推送到 main（验证）
   - [ ] 创建 tag
   - [ ] 推送 tag
   - [ ] 验证 GitHub Release

## 故障排除

### 常见问题

1. **版本获取失败**：
   - 检查是否在正确的目录运行
   - 确认 git 仓库状态正常
   - 检查 setuptools_scm 配置

2. **构建失败**：
   - 检查 Python 版本（需要 3.10+）
   - 确认依赖已正确安装
   - 查看构建日志定位问题

3. **权限错误**：
   - 确认 GITHUB_TOKEN 有 release 权限
   - 检查仓库设置

## 联系方式

如有问题，请联系：
- 项目维护者：EdwinZhanCN
- 邮箱：support@lumilio.org
