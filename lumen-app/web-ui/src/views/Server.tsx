import { useState, useEffect } from "react";
import {
  Play,
  Square,
  Server as ServerIcon,
  CheckCircle,
  RefreshCw,
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
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { WizardLayout } from "@/components/wizard/WizardLayout";
import { useWizard } from "@/context/useWizard";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  getServerStatus,
  startServer,
  stopServer,
  restartServer,
  getServerLogs,
  type ServerStatus,
} from "@/lib/api";

export function Server() {
  const { wizardData, updateWizardData } = useWizard();
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Query server status
  const {
    data: serverStatus,
    refetch: refetchStatus,
    isLoading: statusLoading,
  } = useQuery({
    queryKey: ["serverStatus"],
    queryFn: getServerStatus,
    refetchInterval: autoRefresh ? 2000 : false,
  });

  // Query server logs
  const { data: serverLogs, refetch: refetchLogs } = useQuery({
    queryKey: ["serverLogs"],
    queryFn: () => getServerLogs({ lines: 100 }),
    enabled: serverStatus?.running || false,
    refetchInterval: autoRefresh && serverStatus?.running ? 2000 : false,
  });

  // Start server mutation
  const {
    mutate: startServerMutation,
    isPending: isStarting,
    error: startError,
  } = useMutation({
    mutationFn: startServer,
    onSuccess: () => {
      setAutoRefresh(true);
      updateWizardData({ serverRunning: true });
      refetchStatus();
      setTimeout(() => refetchLogs(), 1000);
    },
  });

  // Stop server mutation
  const {
    mutate: stopServerMutation,
    isPending: isStopping,
    error: stopError,
  } = useMutation({
    mutationFn: stopServer,
    onSuccess: () => {
      updateWizardData({ serverRunning: false });
      refetchStatus();
    },
  });

  // Restart server mutation
  const {
    mutate: restartServerMutation,
    isPending: isRestarting,
    error: restartError,
  } = useMutation({
    mutationFn: restartServer,
    onSuccess: () => {
      setAutoRefresh(true);
      refetchStatus();
      setTimeout(() => refetchLogs(), 1000);
    },
  });

  // Update wizard data when server status changes
  useEffect(() => {
    if (serverStatus) {
      updateWizardData({ serverRunning: serverStatus.running });
    }
  }, [serverStatus, updateWizardData]);

  const handleStartServer = () => {
    startServerMutation({
      config_path: wizardData.configPath,
      port: wizardData.port,
      host: "0.0.0.0",
      environment: "lumen_env",
    });
  };

  const handleStopServer = () => {
    stopServerMutation({
      force: false,
      timeout: 30,
    });
  };

  const handleRestartServer = () => {
    restartServerMutation({
      config_path: wizardData.configPath,
      port: wizardData.port,
      host: "0.0.0.0",
      environment: "lumen_env",
      force: false,
      timeout: 30,
    });
  };

  const getHealthBadge = (health: ServerStatus["health"]) => {
    switch (health) {
      case "healthy":
        return (
          <Badge variant="default" className="bg-green-500">
            健康
          </Badge>
        );
      case "unhealthy":
        return <Badge variant="destructive">异常</Badge>;
      default:
        return <Badge variant="secondary">未知</Badge>;
    }
  };

  const isOperating = isStarting || isStopping || isRestarting;
  const operationError = startError || stopError || restartError;

  return (
    <WizardLayout
      title="启动服务"
      description="配置完成，现在可以启动 Lumen 推理服务器"
      hideNextButton
    >
      <div className="space-y-6">
        {/* Completion Alert */}
        {wizardData.installationComplete && (
          <Alert className="bg-green-50 border-green-200">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">
              所有配置已完成！您可以随时启动或停止服务器。
            </AlertDescription>
          </Alert>
        )}

        {/* Configuration Summary */}
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
                <dd className="font-mono font-semibold">
                  {wizardData.installPath}
                </dd>
              </div>
              <div className="flex justify-between border-b pb-2">
                <dt className="text-muted-foreground font-medium">区域</dt>
                <dd className="font-semibold">
                  {wizardData.region === "cn"
                    ? "中国 (CN)"
                    : "国际 (International)"}
                </dd>
              </div>
              <div className="flex justify-between border-b pb-2">
                <dt className="text-muted-foreground font-medium">服务地址</dt>
                <dd className="font-mono font-semibold">
                  0.0.0.0:{wizardData.port}
                </dd>
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
                    <Badge
                      key={service}
                      variant="secondary"
                      className="text-xs"
                    >
                      {service.toUpperCase()}
                    </Badge>
                  ))}
                </dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        {/* Server Status */}
        <Card>
          <CardHeader>
            <CardTitle>服务器状态</CardTitle>
            <CardDescription>
              {statusLoading ? "正在获取状态..." : "查看服务器实时运行状态"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {statusLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : serverStatus ? (
              <>
                {/* Status Overview */}
                <div className="flex items-center justify-between p-4 rounded-lg border bg-muted/50">
                  <div className="flex items-center gap-3">
                    <div
                      className={`h-3 w-3 rounded-full ${
                        serverStatus.running
                          ? "bg-green-500 animate-pulse"
                          : "bg-red-500"
                      }`}
                    />
                    <div>
                      <p className="font-medium">
                        {serverStatus.running ? "运行中" : "已停止"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {serverStatus.running
                          ? `PID: ${serverStatus.pid} • 运行时间: ${Math.floor((serverStatus.uptime_seconds || 0) / 60)} 分钟`
                          : "服务器未运行"}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getHealthBadge(serverStatus.health)}
                    <Badge
                      variant={serverStatus.running ? "default" : "secondary"}
                      className={serverStatus.running ? "bg-green-500" : ""}
                    >
                      {serverStatus.running ? "运行中" : "已停止"}
                    </Badge>
                  </div>
                </div>

                {/* Server Details */}
                {serverStatus.running && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">监听地址</p>
                      <p className="font-mono font-medium">
                        {serverStatus.host}:{serverStatus.port}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">服务名称</p>
                      <p className="font-medium">{serverStatus.service_name}</p>
                    </div>
                    {serverStatus.config_path && (
                      <div className="col-span-2">
                        <p className="text-muted-foreground">配置文件</p>
                        <p className="font-mono text-xs truncate">
                          {serverStatus.config_path}
                        </p>
                      </div>
                    )}
                    <div>
                      <p className="text-muted-foreground">环境</p>
                      <p className="font-medium">{serverStatus.environment}</p>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {serverStatus.last_error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      {serverStatus.last_error}
                    </AlertDescription>
                  </Alert>
                )}
              </>
            ) : (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>无法获取服务器状态</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Server Control */}
        <Card>
          <CardHeader>
            <CardTitle>服务器控制</CardTitle>
            <CardDescription>启动、停止或重启 Lumen 推理服务器</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Operation Error */}
            {operationError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {operationError.message || "操作失败"}
                </AlertDescription>
              </Alert>
            )}

            {/* Control Buttons */}
            <div className="flex gap-3">
              {!serverStatus?.running ? (
                <button
                  onClick={handleStartServer}
                  disabled={isOperating || !wizardData.installationComplete}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isStarting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  {isStarting ? "启动中..." : "启动服务器"}
                </button>
              ) : (
                <>
                  <button
                    onClick={handleStopServer}
                    disabled={isOperating}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-destructive text-destructive-foreground rounded-md hover:bg-destructive/90 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isStopping ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Square className="h-4 w-4" />
                    )}
                    {isStopping ? "停止中..." : "停止服务器"}
                  </button>
                  <button
                    onClick={handleRestartServer}
                    disabled={isOperating}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-orange-500 text-white rounded-md hover:bg-orange-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isRestarting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4" />
                    )}
                    {isRestarting ? "重启中..." : "重启服务器"}
                  </button>
                </>
              )}
            </div>

            {/* Auto Refresh Toggle */}
            <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/50">
              <div>
                <p className="text-sm font-medium">自动刷新</p>
                <p className="text-xs text-muted-foreground">
                  每 2 秒更新状态和日志
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Server Logs */}
        {serverStatus?.running && serverLogs && serverLogs.logs && (
          <Card>
            <CardHeader>
              <CardTitle>服务器日志</CardTitle>
              <CardDescription>
                实时服务器输出 (最近 {serverLogs.logs.length} 行 / 共{" "}
                {serverLogs.total_lines} 行)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md bg-muted p-4 font-mono text-xs max-h-96 overflow-y-auto">
                {serverLogs.logs.length > 0 ? (
                  serverLogs.logs.map((log, idx) => (
                    <div key={idx} className="text-muted-foreground">
                      {log}
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground text-center py-4">
                    暂无日志
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Usage Guide */}
        <Card className="bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-blue-900">使用指南</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-blue-800 space-y-2">
            <p>服务器启动后，您可以：</p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>
                使用 gRPC 客户端连接到{" "}
                <code className="bg-blue-100 px-1 py-0.5 rounded">
                  0.0.0.0:{wizardData.port}
                </code>
              </li>
              <li>
                通过 mDNS 服务发现查找服务{" "}
                <code className="bg-blue-100 px-1 py-0.5 rounded">
                  {wizardData.serviceName}
                </code>
              </li>
              <li>调用已启用的 AI 服务接口进行推理</li>
              <li>查看实时日志监控服务状态</li>
              <li>使用重启功能更新配置而无需停止服务</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </WizardLayout>
  );
}
