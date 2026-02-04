import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FolderOpen,
  Info,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ArrowRight,
  Settings,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { WizardLayout } from "@/components/wizard/WizardLayout";
import { useWizard } from "@/context/useWizard";
import type { Region } from "@/context/wizardConfig";
import { useMutation } from "@tanstack/react-query";
import {
  validatePath,
  checkInstallationPath,
  type PathValidationResponse,
  type CheckInstallationPathResponse,
} from "@/lib/api";

type ValidationState =
  | { status: "idle" }
  | { status: "validating" }
  | {
      status: "validated";
      pathResult: PathValidationResponse;
      installResult?: CheckInstallationPathResponse;
    }
  | { status: "error"; message: string };

export function Welcome() {
  const { wizardData, updateWizardData } = useWizard();
  const navigate = useNavigate();
  const [installPath, setInstallPath] = useState(wizardData.installPath);
  const [region, setRegion] = useState<Region>(wizardData.region);
  const [serviceName, setServiceName] = useState(wizardData.serviceName);
  const [port, setPort] = useState(wizardData.port.toString());
  const [validationState, setValidationState] = useState<ValidationState>({
    status: "idle",
  });

  // Path validation mutation
  const validatePathMutation = useMutation({
    mutationFn: validatePath,
  });

  // Installation check mutation
  const checkInstallationMutation = useMutation({
    mutationFn: checkInstallationPath,
  });

  // Handle validate button click
  const handleValidate = async () => {
    if (!installPath.trim()) {
      setValidationState({ status: "error", message: "请输入安装路径" });
      return;
    }

    setValidationState({ status: "validating" });

    try {
      // Step 1: Validate path
      const pathResult = await validatePathMutation.mutateAsync({
        path: installPath,
      });

      if (!pathResult.valid) {
        setValidationState({
          status: "validated",
          pathResult,
        });
        return;
      }

      // Step 2: Check installation status
      const installResult =
        await checkInstallationMutation.mutateAsync(installPath);

      // Update wizard data
      updateWizardData({
        installPath,
        region,
        serviceName,
        port: parseInt(port) || 50051,
      });

      // If existing installation is ready, navigate directly to server page
      if (
        installResult.ready_to_start &&
        installResult.recommended_action === "start_existing"
      ) {
        // Navigate to independent server page with path parameter
        navigate(`/server?path=${encodeURIComponent(installPath)}`);
        return;
      }

      // Otherwise, show configuration
      setValidationState({
        status: "validated",
        pathResult,
        installResult,
      });
    } catch (error) {
      setValidationState({
        status: "error",
        message: error instanceof Error ? error.message : "验证失败，请重试",
      });
    }
  };

  // Update wizard data when config changes (only when showing config)
  const handleConfigChange = (updates: {
    region?: Region;
    serviceName?: string;
    port?: string;
  }) => {
    if (updates.region !== undefined) setRegion(updates.region);
    if (updates.serviceName !== undefined) setServiceName(updates.serviceName);
    if (updates.port !== undefined) setPort(updates.port);

    updateWizardData({
      installPath,
      region: updates.region ?? region,
      serviceName: updates.serviceName ?? serviceName,
      port: parseInt(updates.port ?? port) || 50051,
    });
  };

  const isValidating = validationState.status === "validating";
  const isValidated = validationState.status === "validated";
  const pathIsValid = isValidated && validationState.pathResult.valid === true;
  const showConfig = pathIsValid;

  return (
    <WizardLayout
      title="欢迎使用 Lumen"
      description="让我们开始配置您的分布式推理系统"
      hidePrevButton
    >
      <div className="space-y-6">
        {/* Info Alert */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            请填写服务器上的安装路径。Lumen 将在此目录下创建环境并下载模型文件。
          </AlertDescription>
        </Alert>

        {/* Installation Path */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="h-5 w-5" />
              安装路径
            </CardTitle>
            <CardDescription>
              指定 Lumen 的工作目录，用于存储模型、缓存和配置文件
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="installPath">安装路径 *</Label>
              <div className="flex gap-2">
                <Input
                  id="installPath"
                  placeholder="~/.lumen 或 /opt/lumen"
                  value={installPath}
                  onChange={(e) => {
                    setInstallPath(e.target.value);
                    // Reset validation when path changes
                    if (validationState.status !== "idle") {
                      setValidationState({ status: "idle" });
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleValidate();
                    }
                  }}
                  className={`font-mono flex-1 ${
                    isValidated && validationState.pathResult.valid
                      ? "border-green-500"
                      : isValidated && validationState.pathResult.error
                        ? "border-red-500"
                        : validationState.status === "error"
                          ? "border-red-500"
                          : ""
                  }`}
                  disabled={isValidating}
                />
                <Button
                  onClick={handleValidate}
                  disabled={isValidating || !installPath.trim()}
                  className="shrink-0"
                >
                  {isValidating ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ArrowRight className="h-4 w-4" />
                  )}
                </Button>
              </div>

              {/* Error state */}
              {validationState.status === "error" && (
                <Alert variant="destructive" className="mt-2">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{validationState.message}</AlertDescription>
                </Alert>
              )}

              {/* Path Validation Messages */}
              {isValidated && validationState.pathResult.error && (
                <Alert variant="destructive" className="mt-2">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    {validationState.pathResult.error}
                  </AlertDescription>
                </Alert>
              )}

              {isValidated && validationState.pathResult.warning && (
                <Alert className="mt-2 border-yellow-500 bg-yellow-50">
                  <Info className="h-4 w-4 text-yellow-600" />
                  <AlertDescription className="text-yellow-800">
                    {validationState.pathResult.warning}
                  </AlertDescription>
                </Alert>
              )}

              {isValidated && validationState.pathResult.valid && (
                <Alert className="mt-2 border-green-500 bg-green-50">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-800">
                    路径有效 • 可用空间:{" "}
                    {validationState.pathResult.free_space_gb?.toFixed(1)} GB
                    {validationState.pathResult.exists
                      ? " • 目录已存在"
                      : " • 将创建新目录"}
                  </AlertDescription>
                </Alert>
              )}

              {/* Installation Status Message */}
              {isValidated && validationState.installResult?.message && (
                <Alert className="mt-2 border-gray-200">
                  <Info className="h-4 w-4 text-gray-500" />
                  <AlertDescription>
                    {validationState.installResult.message}
                  </AlertDescription>
                </Alert>
              )}

              <p className="text-xs text-muted-foreground">
                建议使用绝对路径，确保有足够的磁盘空间（至少 10GB）
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Server Configuration - Only show after path validation */}
        {showConfig && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                服务器配置
              </CardTitle>
              <CardDescription>配置 gRPC 服务器的基本参数</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Region */}
                <div className="space-y-2">
                  <Label htmlFor="region">区域 *</Label>
                  <Select
                    value={region}
                    onValueChange={(value: string) =>
                      handleConfigChange({ region: value as Region })
                    }
                  >
                    <SelectTrigger id="region">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cn">中国 (CN)</SelectItem>
                      <SelectItem value="other">
                        国际 (International)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    影响模型下载源和优化策略
                  </p>
                </div>

                {/* Port */}
                <div className="space-y-2">
                  <Label htmlFor="port">端口号 *</Label>
                  <Input
                    id="port"
                    type="number"
                    min="1024"
                    max="65535"
                    value={port}
                    onChange={(e) =>
                      handleConfigChange({ port: e.target.value })
                    }
                  />
                  <p className="text-xs text-muted-foreground">
                    gRPC 服务监听端口（默认 50051）
                  </p>
                </div>
              </div>

              {/* Service Name */}
              <div className="space-y-2">
                <Label htmlFor="serviceName">服务名称 *</Label>
                <Input
                  id="serviceName"
                  placeholder="lumen-server"
                  value={serviceName}
                  onChange={(e) =>
                    handleConfigChange({ serviceName: e.target.value })
                  }
                />
                <p className="text-xs text-muted-foreground">
                  用于 mDNS 服务发现的名称
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Summary - Only show after path validation */}
        {showConfig && (
          <Card className="bg-muted/50">
            <CardHeader>
              <CardTitle className="text-base">配置摘要</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">安装路径:</dt>
                  <dd className="font-mono font-medium">
                    {installPath || "未设置"}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">区域:</dt>
                  <dd className="font-medium">
                    {region === "cn" ? "中国 (CN)" : "国际 (International)"}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">服务地址:</dt>
                  <dd className="font-mono font-medium">0.0.0.0:{port}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">服务名称:</dt>
                  <dd className="font-medium">{serviceName}</dd>
                </div>
              </dl>
            </CardContent>
          </Card>
        )}
      </div>
    </WizardLayout>
  );
}
