import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Play,
  Square,
  Server as ServerIcon,
  RefreshCw,
  AlertCircle,
  Loader2,
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
import { useLumenSession } from "@/hooks/useLumenSession";
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
  const navigate = useNavigate();
  const { currentPath } = useLumenSession();
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [showYamlDialog, setShowYamlDialog] = useState(false);
  const [yamlContent, setYamlContent] = useState("");
  const [copied, setCopied] = useState(false);

  const loadConfigMutation = useMutation({
    mutationFn: (configPath: string) => loadConfig(configPath),
  });

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

  useEffect(() => {
    if (!currentPath) {
      return;
    }

    const configPath = `${currentPath}/lumen-config.yaml`;
    loadConfigMutation.mutate(configPath, {
      onSuccess: () => {
        refetchConfig();
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPath]);

  const loadedConfig = currentConfig?.loaded ? currentConfig : null;
  const installPath = currentPath || loadedConfig?.cache_dir || "~/.lumen";
  const configPath = loadedConfig?.config_path || undefined;
  const envName = loadedConfig?.env_name || "lumen_env";
  const port = loadedConfig?.port || 50051;

  const hasValidConfig = !!configPath;
  const isConfigLoaded = !configLoading && loadedConfig !== null;

  const {
    data: serverStatus,
    refetch: refetchStatus,
    isLoading: statusLoading,
  } = useQuery({
    queryKey: ["serverStatus"],
    queryFn: getServerStatus,
    refetchInterval: autoRefresh ? 2000 : false,
  });

  const { data: serverLogs, refetch: refetchLogs } = useQuery({
    queryKey: ["serverLogs"],
    queryFn: () => getServerLogs({ lines: 100 }),
    enabled: serverStatus?.running || false,
    refetchInterval: autoRefresh && serverStatus?.running ? 2000 : false,
  });

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
      port,
      host: "0.0.0.0",
      environment: envName,
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
      port,
      host: "0.0.0.0",
      environment: envName,
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

  const handleCopyYaml = async () => {
    try {
      await navigator.clipboard.writeText(yamlContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const showSetupPrompt = !configLoading && !hasValidConfig;

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold">
            <ServerIcon className="h-5 w-5" />
            Lumen 服务管理
          </h1>
          <p className="text-sm text-muted-foreground">管理和监控 Lumen 推理服务器</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleViewYaml}
            disabled={!hasValidConfig}
          >
            <FileText className="mr-2 h-4 w-4" />
            查看配置
          </Button>
          <Button variant="outline" size="sm" onClick={() => navigate("/session")}>
            <Settings className="mr-2 h-4 w-4" />
            重新配置
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="space-y-6 lg:col-span-2">
          {showSetupPrompt && (
            <Alert className="border-yellow-500 bg-yellow-50">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="text-yellow-800">
                <div className="flex items-center justify-between">
                  <span>未检测到有效配置，请先完成会话入口中的配置流程。</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => navigate("/session")}
                    className="ml-4 border-yellow-600 text-yellow-700 hover:bg-yellow-100"
                  >
                    前往会话入口
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          )}

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
                      <dd className="truncate font-mono text-xs">{configPath}</dd>
                    </div>
                  )}
                </dl>
              ) : (
                <p className="text-sm text-muted-foreground">
                  暂无配置信息，请从会话入口重新进入配置流程。
                </p>
              )}
            </CardContent>
          </Card>

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
                  <div className="flex items-center justify-between rounded-lg border bg-muted/50 p-4">
                    <div className="flex items-center gap-3">
                      <div
                        className={`h-3 w-3 rounded-full ${
                          serverStatus.running ? "bg-green-500 animate-pulse" : "bg-red-500"
                        }`}
                      />
                      <div>
                        <p className="font-medium">{serverStatus.running ? "运行中" : "已停止"}</p>
                        <p className="text-xs text-muted-foreground">
                          {serverStatus.running
                            ? `PID: ${serverStatus.pid} • 运行时间: ${Math.floor((serverStatus.uptime_seconds || 0) / 60)} 分钟`
                            : "服务器未运行"}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {serverStatus.running && getHealthBadge(serverStatus.health)}
                      <Badge
                        variant={serverStatus.running ? "default" : "secondary"}
                        className={serverStatus.running ? "bg-green-500" : ""}
                      >
                        {serverStatus.running ? "运行中" : "已停止"}
                      </Badge>
                    </div>
                  </div>

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
                          <p className="truncate font-mono text-xs">
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

                  {serverStatus.last_error && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{serverStatus.last_error}</AlertDescription>
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

          <Card>
            <CardHeader>
              <CardTitle>服务器控制</CardTitle>
              <CardDescription>启动、停止或重启 Lumen 推理服务器</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {operationError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{operationError.message || "操作失败"}</AlertDescription>
                </Alert>
              )}

              <div className="flex gap-3">
                {!serverStatus?.running ? (
                  <Button
                    onClick={handleStartServer}
                    disabled={isOperating || !hasValidConfig}
                    className="flex-1"
                    size="lg"
                  >
                    {isStarting ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="mr-2 h-4 w-4" />
                    )}
                    {isStarting ? "启动中..." : !hasValidConfig ? "需要配置" : "启动服务器"}
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
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Square className="mr-2 h-4 w-4" />
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
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="mr-2 h-4 w-4" />
                      )}
                      {isRestarting ? "重启中..." : "重启服务器"}
                    </Button>
                  </>
                )}
              </div>

              <div className="flex items-center justify-between rounded-lg border bg-muted/50 p-3">
                <div>
                  <p className="text-sm font-medium">自动刷新</p>
                  <p className="text-xs text-muted-foreground">每 2 秒更新状态和日志</p>
                </div>
                <label className="relative inline-flex cursor-pointer items-center">
                  <input
                    type="checkbox"
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                    className="peer sr-only"
                  />
                  <div className="peer h-6 w-11 rounded-full bg-gray-200 after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5 after:rounded-full after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:bg-primary peer-checked:after:translate-x-full peer-checked:after:border-white peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20"></div>
                </label>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-3">
          {serverStatus?.running && serverLogs && serverLogs.logs ? (
            <Card className="sticky top-20">
              <CardHeader>
                <CardTitle className="text-base">服务器日志</CardTitle>
                <CardDescription className="text-xs">
                  最近 {serverLogs.logs.length} 行 / 共 {serverLogs.total_lines} 行
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[calc(100vh-16rem)] overflow-y-auto rounded-md bg-muted p-3 font-mono text-xs">
                  {serverLogs.logs.length > 0 ? (
                    serverLogs.logs.map((log, idx) => (
                      <div key={idx} className="mb-1 text-muted-foreground">
                        {log}
                      </div>
                    ))
                  ) : (
                    <div className="py-4 text-center text-muted-foreground">暂无日志</div>
                  )}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="sticky top-20">
              <CardHeader>
                <CardTitle className="text-base">服务器日志</CardTitle>
                <CardDescription className="text-xs">服务器运行时显示日志</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-muted p-4 text-center text-sm text-muted-foreground">
                  服务器未运行
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {showYamlDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="flex max-h-[80vh] w-full max-w-4xl flex-col rounded-lg border bg-background text-foreground shadow-xl">
            <div className="flex items-center justify-between border-b p-4">
              <h2 className="flex items-center gap-2 text-lg font-semibold">
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
                      <Check className="mr-2 h-4 w-4" />
                      已复制
                    </>
                  ) : (
                    <>
                      <Copy className="mr-2 h-4 w-4" />
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

            <div className="flex-1 overflow-auto p-4">
              <pre className="overflow-x-auto rounded-md bg-muted p-4 text-xs font-mono">
                {yamlContent}
              </pre>
            </div>

            <div className="flex justify-end gap-2 border-t p-4">
              <Button variant="outline" onClick={() => setShowYamlDialog(false)}>
                关闭
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
