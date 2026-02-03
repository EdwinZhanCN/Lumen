import { useState } from "react";
import { Play, Square, Server as ServerIcon, CheckCircle } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { WizardLayout } from "@/components/wizard/WizardLayout";
import { useWizard } from "@/context/WizardContext";

export function Server() {
  const { wizardData, updateWizardData } = useWizard();
  const [serverRunning, setServerRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);

  const handleStartServer = () => {
    setServerRunning(true);
    updateWizardData({ serverRunning: true });
    setLogs([
      `[${new Date().toLocaleTimeString()}] 🚀 启动 Lumen 服务器...`,
      `[${new Date().toLocaleTimeString()}] 📍 安装路径: ${wizardData.installPath}`,
      `[${new Date().toLocaleTimeString()}] 🌐 监听地址: 0.0.0.0:${wizardData.port}`,
      `[${new Date().toLocaleTimeString()}] 📦 启用服务: ${wizardData.selectedServices.join(", ")}`,
      `[${new Date().toLocaleTimeString()}] ✓ gRPC 服务器启动成功`,
      `[${new Date().toLocaleTimeString()}] ✓ mDNS 服务发现已启用 (${wizardData.serviceName})`,
      `[${new Date().toLocaleTimeString()}] 🎉 服务器运行中，等待连接...`,
    ]);
  };

  const handleStopServer = () => {
    setServerRunning(false);
    updateWizardData({ serverRunning: false });
    setLogs((prev) => [
      ...prev,
      `[${new Date().toLocaleTimeString()}] 🛑 正在停止服务器...`,
      `[${new Date().toLocaleTimeString()}] ✓ 服务器已停止`,
    ]);
  };

  return (
    <WizardLayout
      title="启动服务"
      description="配置完成，现在可以启动 Lumen 推理服务器"
      hideNextButton
    >
      <div className="space-y-6">
        <Alert className="bg-green-50 border-green-200">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            所有配置已完成！您可以随时启动或停止服务器。
          </AlertDescription>
        </Alert>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ServerIcon className="h-5 w-5" />
              配置摘要
            </CardTitle>
            <CardDescription>查看您的完整配置信息</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between border-b pb-2">
                <dt className="text-muted-foreground font-medium">安装路径</dt>
                <dd className="font-mono font-semibold">{wizardData.installPath}</dd>
              </div>
              <div className="flex justify-between border-b pb-2">
                <dt className="text-muted-foreground font-medium">区域</dt>
                <dd className="font-semibold">{wizardData.region === "cn" ? "中国" : "国际"}</dd>
              </div>
              <div className="flex justify-between border-b pb-2">
                <dt className="text-muted-foreground font-medium">服务地址</dt>
                <dd className="font-mono font-semibold">0.0.0.0:{wizardData.port}</dd>
              </div>
              <div className="flex justify-between border-b pb-2">
                <dt className="text-muted-foreground font-medium">服务名称</dt>
                <dd className="font-semibold">{wizardData.serviceName}</dd>
              </div>
              <div className="flex justify-between border-b pb-2">
                <dt className="text-muted-foreground font-medium">硬件预设</dt>
                <dd className="font-semibold">{wizardData.hardwarePreset}</dd>
              </div>
              <div className="flex justify-between items-start">
                <dt className="text-muted-foreground font-medium">启用服务</dt>
                <dd className="flex flex-wrap gap-1 justify-end max-w-xs">
                  {wizardData.selectedServices.map((service) => (
                    <Badge key={service} variant="secondary" className="text-xs">
                      {service.toUpperCase()}
                    </Badge>
                  ))}
                </dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>服务器控制</CardTitle>
            <CardDescription>启动或停止 Lumen 推理服务器</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-lg border bg-muted/50">
              <div className="flex items-center gap-3">
                <div className={`h-3 w-3 rounded-full ${serverRunning ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
                <div>
                  <p className="font-medium">{serverRunning ? "运行中" : "已停止"}</p>
                  <p className="text-xs text-muted-foreground">
                    {serverRunning ? "服务器正在运行，可接受请求" : "服务器未运行"}
                  </p>
                </div>
              </div>
              <Badge variant={serverRunning ? "default" : "secondary"} className={serverRunning ? "bg-green-500" : ""}>
                {serverRunning ? "运行中" : "已停止"}
              </Badge>
            </div>

            <div className="flex gap-3">
              {!serverRunning ? (
                <button
                  onClick={handleStartServer}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors font-medium"
                >
                  <Play className="h-4 w-4" />
                  启动服务器
                </button>
              ) : (
                <button
                  onClick={handleStopServer}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-destructive text-destructive-foreground rounded-md hover:bg-destructive/90 transition-colors font-medium"
                >
                  <Square className="h-4 w-4" />
                  停止服务器
                </button>
              )}
            </div>
          </CardContent>
        </Card>

        {logs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>服务器日志</CardTitle>
              <CardDescription>实时服务器输出</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md bg-muted p-4 font-mono text-xs max-h-96 overflow-y-auto">
                {logs.map((log, idx) => (
                  <div key={idx} className="text-muted-foreground">{log}</div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Card className="bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-blue-900">下一步</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-blue-800 space-y-2">
            <p>服务器启动后，您可以：</p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>使用 gRPC 客户端连接到 <code className="bg-blue-100 px-1 py-0.5 rounded">0.0.0.0:{wizardData.port}</code></li>
              <li>通过 mDNS 服务发现查找服务 <code className="bg-blue-100 px-1 py-0.5 rounded">{wizardData.serviceName}</code></li>
              <li>调用已启用的 AI 服务接口进行推理</li>
              <li>查看实时日志监控服务状态</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </WizardLayout>
  );
}
