import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Download,
  CheckCircle,
  Loader2,
  AlertCircle,
  Play,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { WizardLayout } from "@/components/wizard/WizardLayout";
import { useWizard } from "@/context/useWizard";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  getInstallStatus,
  startInstallation,
  getInstallTask,
  getInstallLogs,
  loadConfig,
  type InstallTaskResponse,
  type InstallStep,
} from "@/lib/api";

export function Install() {
  const { wizardData, updateWizardData } = useWizard();
  const navigate = useNavigate();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [isFinishing, setIsFinishing] = useState(false);
  const [finishError, setFinishError] = useState<string | null>(null);

  // Query install status
  const { data: installStatus } = useQuery({
    queryKey: ["installStatus"],
    queryFn: getInstallStatus,
  });

  // Start installation mutation
  const {
    mutate: startInstall,
    isPending: isStarting,
    error: startError,
  } = useMutation({
    mutationFn: startInstallation,
    onSuccess: (data) => {
      setTaskId(data.task_id);
    },
  });

  // Poll task status
  const { data: taskStatus } = useQuery({
    queryKey: ["installTask", taskId],
    queryFn: () => getInstallTask(taskId!),
    enabled: taskId !== null,
    refetchInterval: (query) => {
      const data = query.state.data as InstallTaskResponse | undefined;
      // Stop polling if completed or failed
      if (data?.status === "completed" || data?.status === "failed") {
        return false;
      }
      return 1000; // Poll every second
    },
  });

  // Query task logs
  const { data: taskLogs } = useQuery({
    queryKey: ["installLogs", taskId],
    queryFn: () => getInstallLogs(taskId!, { tail: 100 }),
    enabled:
      taskId !== null &&
      (taskStatus?.status === "running" || taskStatus?.status === "pending"),
    refetchInterval: 2000, // Refresh logs every 2 seconds
  });

  // Update wizard data when task is complete
  useEffect(() => {
    if (taskStatus?.status === "completed") {
      updateWizardData({ installationComplete: true });
    }
  }, [taskStatus?.status, updateWizardData]);

  const handleStartInstall = () => {
    if (!wizardData.hardwarePreset) {
      return;
    }

    startInstall({
      preset: wizardData.hardwarePreset,
      cache_dir: wizardData.installPath || "~/.lumen",
      environment_name: "lumen_env",
      force_reinstall: false,
    });
  };

  const handleFinishInstall = async () => {
    setFinishError(null);
    setIsFinishing(true);
    try {
      const configPath =
        wizardData.configPath ||
        (wizardData.installPath
          ? `${wizardData.installPath}/lumen-config.yaml`
          : undefined);

      if (!configPath) {
        setFinishError("未找到配置文件路径，请返回上一步重新生成配置。");
        return;
      }

      await loadConfig(configPath);
      navigate("/server");
    } catch (error) {
      setFinishError(
        error instanceof Error ? error.message : "完成步骤失败，请重试。",
      );
    } finally {
      setIsFinishing(false);
    }
  };

  const getStatusIcon = (status: InstallStep["status"]) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "running":
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case "failed":
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case "skipped":
        return (
          <div className="h-5 w-5 rounded-full border-2 border-muted-foreground" />
        );
      default:
        return <div className="h-5 w-5 rounded-full border-2 border-muted" />;
    }
  };

  const getStatusBadge = (status: InstallStep["status"]) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="default" className="bg-green-500">
            完成
          </Badge>
        );
      case "running":
        return <Badge variant="secondary">进行中</Badge>;
      case "failed":
        return <Badge variant="destructive">错误</Badge>;
      case "skipped":
        return <Badge variant="outline">跳过</Badge>;
      default:
        return <Badge variant="outline">等待</Badge>;
    }
  };

  const allCompleted = taskStatus?.status === "completed";
  const hasFailed = taskStatus?.status === "failed";
  const isInstalling = taskStatus?.status === "running" || isStarting;

  return (
    <WizardLayout
      title="安装依赖"
      description="下载并安装所需的运行环境和模型文件"
      onFinish={handleFinishInstall}
      nextButtonDisabled={isFinishing || !allCompleted}
    >
      <div className="space-y-6">
        {/* Installation Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Download className="h-5 w-5" />
              安装进度
            </CardTitle>
            <CardDescription>
              {isInstalling
                ? "正在安装，请稍候..."
                : allCompleted
                  ? "所有任务已完成！"
                  : hasFailed
                    ? "安装失败，请查看错误信息"
                    : "点击开始按钮开始安装"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Overall Progress */}
            {taskStatus && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">总体进度</span>
                  <span className="text-muted-foreground">
                    {taskStatus.progress}%
                  </span>
                </div>
                <Progress value={taskStatus.progress} className="h-2" />
                {taskStatus.current_step && (
                  <p className="text-sm text-muted-foreground">
                    当前步骤: {taskStatus.current_step}
                  </p>
                )}
              </div>
            )}

            {/* Installation Info */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">安装路径</p>
                <p className="font-mono font-medium truncate">
                  {wizardData.installPath}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">硬件预设</p>
                <p className="font-medium">
                  {wizardData.hardwarePreset || "未选择"}
                </p>
              </div>
            </div>

            {/* System Status */}
            {installStatus && (
              <div className="rounded-lg border p-3 bg-muted/50">
                <p className="text-sm font-medium mb-2">系统状态</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    {installStatus.micromamba_installed ? (
                      <CheckCircle className="h-3 w-3 text-green-500" />
                    ) : (
                      <AlertCircle className="h-3 w-3 text-yellow-500" />
                    )}
                    <span>Micromamba</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {installStatus.environment_exists ? (
                      <CheckCircle className="h-3 w-3 text-green-500" />
                    ) : (
                      <AlertCircle className="h-3 w-3 text-yellow-500" />
                    )}
                    <span>Python 环境</span>
                  </div>
                </div>
                {installStatus.missing_components &&
                  installStatus.missing_components.length > 0 && (
                    <p className="text-xs text-muted-foreground mt-2">
                      缺少组件: {installStatus.missing_components.join(", ")}
                    </p>
                  )}
              </div>
            )}

            {/* Start Button */}
            {!isInstalling && !allCompleted && !hasFailed && (
              <button
                onClick={handleStartInstall}
                disabled={!wizardData.hardwarePreset}
                className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="h-4 w-4" />
                开始安装
              </button>
            )}

            {/* Error Alert */}
            {(startError || hasFailed) && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {startError?.message || taskStatus?.error || "安装失败"}
                </AlertDescription>
              </Alert>
            )}

            {/* Success Alert */}
            {allCompleted && (
              <Alert className="bg-green-50 border-green-200">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-800">
                  安装成功！您可以继续下一步启动服务。
                </AlertDescription>
              </Alert>
            )}

            {finishError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{finishError}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Task Steps */}
        {taskStatus?.steps && taskStatus.steps.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>安装任务</CardTitle>
              <CardDescription>各项安装任务的详细状态</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {taskStatus.steps.map((step, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between rounded-lg border p-4"
                >
                  <div className="flex items-center gap-3 flex-1">
                    {getStatusIcon(step.status)}
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{step.name}</p>
                        {getStatusBadge(step.status)}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {step.message}
                      </p>
                      {step.status === "running" && step.progress > 0 && (
                        <div className="mt-2">
                          <Progress value={step.progress} className="h-1" />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Installation Logs */}
        {taskLogs && taskLogs.logs && taskLogs.logs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>安装日志</CardTitle>
              <CardDescription>
                实时安装过程输出 (最近 {taskLogs.logs.length} 行 / 共{" "}
                {taskLogs.total_lines} 行)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md bg-muted p-4 font-mono text-xs max-h-64 overflow-y-auto">
                {taskLogs.logs.map((log, idx) => (
                  <div key={idx} className="text-muted-foreground">
                    {log}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </WizardLayout>
  );
}
