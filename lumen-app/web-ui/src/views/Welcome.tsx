import { useState, useEffect } from "react";
import {
  FolderOpen,
  Info,
  CheckCircle2,
  AlertCircle,
  Loader2,
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
import { WizardLayout } from "@/components/wizard/WizardLayout";
import { useWizard } from "@/context/useWizard";
import type { Region } from "@/context/wizardConfig";
import { useDebounce } from "@/hooks/useDebounce";
import { useMutation } from "@tanstack/react-query";
import { validatePath } from "@/lib/api.ts";

export function Welcome() {
  const { wizardData, updateWizardData } = useWizard();
  const [installPath, setInstallPath] = useState(wizardData.installPath);
  const [region, setRegion] = useState<Region>(wizardData.region);
  const [serviceName, setServiceName] = useState(wizardData.serviceName);
  const [port, setPort] = useState(wizardData.port.toString());

  const {
    mutate,
    data: pathValidation,
    isPending: validating,
  } = useMutation({ mutationFn: validatePath });

  const debouncedPath = useDebounce(installPath, 500);

  // Validate path when it changes
  useEffect(() => {
    if (!debouncedPath || debouncedPath.trim() === "") {
      return;
    }

    mutate({ path: debouncedPath });
  }, [debouncedPath, mutate]);

  // Update context when local state changes
  useEffect(() => {
    updateWizardData({
      installPath,
      region,
      serviceName,
      port: parseInt(port) || 50051,
    });
  }, [installPath, region, serviceName, port, updateWizardData]);

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
              <div className="relative">
                <Input
                  id="installPath"
                  placeholder="~/.lumen 或 /opt/lumen"
                  value={installPath}
                  onChange={(e) => setInstallPath(e.target.value)}
                  className={`font-mono ${
                    pathValidation?.valid
                      ? "border-green-500"
                      : pathValidation?.error
                        ? "border-red-500"
                        : ""
                  }`}
                />
                {validating && (
                  <Loader2 className="absolute right-3 top-3 h-4 w-4 animate-spin text-muted-foreground" />
                )}
                {!validating && pathValidation?.valid && (
                  <CheckCircle2 className="absolute right-3 top-3 h-4 w-4 text-green-500" />
                )}
                {!validating && pathValidation?.error && (
                  <AlertCircle className="absolute right-3 top-3 h-4 w-4 text-red-500" />
                )}
              </div>

              {/* Path Validation Messages */}
              {pathValidation?.error && (
                <Alert variant="destructive" className="mt-2">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{pathValidation.error}</AlertDescription>
                </Alert>
              )}

              {pathValidation?.warning && (
                <Alert className="mt-2 border-yellow-500 bg-yellow-50">
                  <Info className="h-4 w-4 text-yellow-600" />
                  <AlertDescription className="text-yellow-800">
                    {pathValidation.warning}
                  </AlertDescription>
                </Alert>
              )}

              {pathValidation?.valid && (
                <Alert className="mt-2 border-green-500 bg-green-50">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-800">
                    路径有效 • 可用空间:{" "}
                    {pathValidation.free_space_gb?.toFixed(1)} GB
                    {pathValidation.exists
                      ? " • 目录已存在"
                      : " • 将创建新目录"}
                  </AlertDescription>
                </Alert>
              )}

              <p className="text-xs text-muted-foreground">
                建议使用绝对路径，确保有足够的磁盘空间（至少 10GB）
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Server Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>服务器配置</CardTitle>
            <CardDescription>配置 gRPC 服务器的基本参数</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Region */}
              <div className="space-y-2">
                <Label htmlFor="region">区域 *</Label>
                <Select
                  value={region}
                  onValueChange={(value: string) => setRegion(value as Region)}
                >
                  <SelectTrigger id="region">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cn">中国 (CN)</SelectItem>
                    <SelectItem value="other">国际 (International)</SelectItem>
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
                  onChange={(e) => setPort(e.target.value)}
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
                onChange={(e) => setServiceName(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                用于 mDNS 服务发现的名称
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Summary */}
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
      </div>
    </WizardLayout>
  );
}
