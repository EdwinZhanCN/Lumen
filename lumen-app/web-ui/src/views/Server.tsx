import { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import {
  Play,
  Square,
  Server as ServerIcon,
  RefreshCw,
  AlertCircle,
  Loader2,
  ArrowLeft,
  Settings,
  FileText,
  Copy,
  Check,
  X,
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
import { Button } from "@/components/ui/button";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  getServerStatus,
  startServer,
  stopServer,
  restartServer,
  getServerLogs,
  getCurrentConfig,
  loadConfig,
  getConfigYaml,
  type ServerStatus,
  type CurrentConfigResponse,
} from "@/lib/api";

export function Server() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [showYamlDialog, setShowYamlDialog] = useState(false);
  const [yamlContent, setYamlContent] = useState("");
  const [copied, setCopied] = useState(false);

  // Get install path from URL params (passed from Welcome page)
  const installPathFromUrl = searchParams.get("path");

  // Load config mutation (call once when path is provided)
  const loadConfigMutation = useMutation({
    mutationFn: (configPath: string) => loadConfig(configPath),
  });

  // Query current config from app_state
  const {
    data: currentConfig,
    isLoading: configLoading,
    refetch: refetchConfig,
  } = useQuery<CurrentConfigResponse>({
    queryKey: ["currentConfig"],
    queryFn: getCurrentConfig,
    staleTime: 60000,
    enabled: !loadConfigMutation.isPending,
  });

  // Load config when URL has path parameter (only once)
  useEffect(() => {
    if (installPathFromUrl) {
      const configPath = `${installPathFromUrl}/lumen-config.yaml`;
      loadConfigMutation.mutate(configPath, {
        onSuccess: () => {
          refetchConfig();
        },
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [installPathFromUrl]);

  // Derive config values (prefer URL params, fallback to API config)
  const loadedConfig = currentConfig?.loaded ? currentConfig : null;
  const installPath =
    installPathFromUrl || loadedConfig?.cache_dir || "~/.lumen";
  // Config path: prefer URL param path, fallback to API config
  const configPath = installPathFromUrl
    ? `${installPathFromUrl}/lumen-config.yaml`
    : loadedConfig?.cache_dir
      ? `${loadedConfig.cache_dir}/lumen-config.yaml`
      : undefined;
  const port = loadedConfig?.port || 50051;

  // Check if we have a valid configuration to start the server
  // Valid if we have either URL path param or loaded config from API
  const hasValidConfig = !!(installPathFromUrl || loadedConfig);
  const isConfigLoaded = !configLoading && loadedConfig !== null;

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

  const handleStartServer = () => {
    startServerMutation({
      config_path: configPath,
      port: port,
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
      config_path: configPath,
      port: port,
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

  // Handle view YAML
  const handleViewYaml = async () => {
    try {
      const response = await getConfigYaml();
      if (response.loaded && response.yaml) {
        setYamlContent(response.yaml);
        setShowYamlDialog(true);
      }
    } catch (error) {
      console.error("Failed to load YAML:", error);
    }
  };

  // Handle copy YAML
  const handleCopyYaml = async () => {
    try {
      await navigator.clipboard.writeText(yamlContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  // Show setup prompt if no valid config
  const showSetupPrompt = !configLoading && !hasValidConfig;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/welcome")}
              >
                <ArrowLeft className="h-5 w-5" />
              </Button>
              <div>
                <h1 className="text-xl font-semibold flex items-center gap-2">
                  <ServerIcon className="h-5 w-5" />
                  Lumen 服务管理
                </h1>
                <p className="text-sm text-muted-foreground">
                  管理和监控 Lumen 推理服务器
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleViewYaml}
                disabled={!hasValidConfig}
              >
                <FileText className="h-4 w-4 mr-2" />
                查看配置
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate("/welcome")}
              >
                <Settings className="h-4 w-4 mr-2" />
                重新配置
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Column - Controls and Status */}
          <div className="lg:col-span-2 space-y-6">
            {/* Setup Required Alert */}
            {showSetupPrompt && (
              <Alert className="border-yellow-500 bg-yellow-50">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <AlertDescription className="text-yellow-800">
                  <div className="flex items-center justify-between">
                    <span>
                      未检测到有效配置，请先完成安装向导或指定正确的安装路径。
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate("/welcome")}
                      className="ml-4 border-yellow-600 text-yellow-700 hover:bg-yellow-100"
                    >
                      前往配置
                    </Button>
                  </div>
                </AlertDescription>
              </Alert>
            )}

            {/* Installation Info */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">当前配置</CardTitle>
              </CardHeader>
              <CardContent>
                {configLoading ? (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    加载配置中...
                  </div>
                ) : isConfigLoaded ? (
                  <dl className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <dt className="text-muted-foreground">安装路径</dt>
                      <dd className="font-mono font-medium">{installPath}</dd>
                    </div>
                    <div>
                      <dt className="text-muted-foreground">服务端口</dt>
                      <dd className="font-mono font-medium">0.0.0.0:{port}</dd>
                    </div>
                    {configPath && (
                      <div className="col-span-2">
                        <dt className="text-muted-foreground">配置文件</dt>
                        <dd className="font-mono text-xs truncate">
                          {configPath}
                        </dd>
                      </div>
                    )}
                  </dl>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    暂无配置信息。请通过安装向导完成配置，或确保已正确指定安装路径。
                  </p>
                )}
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
                        {serverStatus.running &&
                          getHealthBadge(serverStatus.health)}
                        <Badge
                          variant={
                            serverStatus.running ? "default" : "secondary"
                          }
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
                          <p className="font-medium">
                            {serverStatus.service_name}
                          </p>
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
                          <p className="font-medium">
                            {serverStatus.environment}
                          </p>
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
                <CardDescription>
                  启动、停止或重启 Lumen 推理服务器
                </CardDescription>
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
                    <Button
                      onClick={handleStartServer}
                      disabled={isOperating || !hasValidConfig}
                      className="flex-1"
                      size="lg"
                    >
                      {isStarting ? (
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      ) : (
                        <Play className="h-4 w-4 mr-2" />
                      )}
                      {isStarting
                        ? "启动中..."
                        : !hasValidConfig
                          ? "需要配置"
                          : "启动服务器"}
                    </Button>
                  ) : (
                    <>
                      <Button
                        onClick={handleStopServer}
                        disabled={isOperating}
                        variant="destructive"
                        className="flex-1"
                        size="lg"
                      >
                        {isStopping ? (
                          <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        ) : (
                          <Square className="h-4 w-4 mr-2" />
                        )}
                        {isStopping ? "停止中..." : "停止服务器"}
                      </Button>
                      <Button
                        onClick={handleRestartServer}
                        disabled={isOperating}
                        variant="outline"
                        className="flex-1 border-orange-500 text-orange-600 hover:bg-orange-50"
                        size="lg"
                      >
                        {isRestarting ? (
                          <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        ) : (
                          <RefreshCw className="h-4 w-4 mr-2" />
                        )}
                        {isRestarting ? "重启中..." : "重启服务器"}
                      </Button>
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
          </div>

          {/* Right Column - Server Logs */}
          <div className="lg:col-span-3">
            {serverStatus?.running && serverLogs && serverLogs.logs ? (
              <Card className="sticky top-6">
                <CardHeader>
                  <CardTitle className="text-base">服务器日志</CardTitle>
                  <CardDescription className="text-xs">
                    最近 {serverLogs.logs.length} 行 / 共{" "}
                    {serverLogs.total_lines} 行
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-md bg-muted p-3 font-mono text-xs h-[calc(100vh-16rem)] overflow-y-auto">
                    {serverLogs.logs.length > 0 ? (
                      serverLogs.logs.map((log, idx) => (
                        <div key={idx} className="text-muted-foreground mb-1">
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
            ) : (
              <Card className="sticky top-6">
                <CardHeader>
                  <CardTitle className="text-base">服务器日志</CardTitle>
                  <CardDescription className="text-xs">
                    服务器运行时显示日志
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-md bg-muted p-4 text-center text-muted-foreground text-sm">
                    服务器未运行
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>

      {/* YAML Dialog */}
      {showYamlDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] flex flex-col">
            {/* Dialog Header */}
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <FileText className="h-5 w-5" />
                配置文件内容
              </h2>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCopyYaml}
                  disabled={copied}
                >
                  {copied ? (
                    <>
                      <Check className="h-4 w-4 mr-2" />
                      已复制
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4 mr-2" />
                      复制
                    </>
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowYamlDialog(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Dialog Content */}
            <div className="flex-1 overflow-auto p-4">
              <pre className="bg-muted p-4 rounded-md text-xs font-mono overflow-x-auto">
                {yamlContent}
              </pre>
            </div>

            {/* Dialog Footer */}
            <div className="flex justify-end gap-2 p-4 border-t">
              <Button
                variant="outline"
                onClick={() => setShowYamlDialog(false)}
              >
                关闭
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
