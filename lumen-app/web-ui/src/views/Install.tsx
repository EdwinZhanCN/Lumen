import { useState } from "react";
import { Download, CheckCircle, Loader2, AlertCircle } from "lucide-react";
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
import { useWizard } from "@/context/WizardContext";

interface InstallTask {
  id: string;
  name: string;
  description: string;
  status: "pending" | "in_progress" | "completed" | "error";
  progress: number;
}

export function Install() {
  const { wizardData, updateWizardData } = useWizard();
  const [installing, setInstalling] = useState(false);
  const [tasks, setTasks] = useState<InstallTask[]>([
    {
      id: "micromamba",
      name: "Micromamba",
      description: "安装包管理器",
      status: "pending",
      progress: 0,
    },
    {
      id: "environment",
      name: "Python 环境",
      description: "创建隔离环境",
      status: "pending",
      progress: 0,
    },
    {
      id: "dependencies",
      name: "依赖库",
      description: "安装 Python 依赖",
      status: "pending",
      progress: 0,
    },
    {
      id: "models",
      name: "模型文件",
      description: "下载 AI 模型",
      status: "pending",
      progress: 0,
    },
  ]);
  const [currentLog, setCurrentLog] = useState<string[]>([]);

  const simulateInstallation = async () => {
    setInstalling(true);
    const updatedTasks = [...tasks];

    for (let i = 0; i < updatedTasks.length; i++) {
      // Start task
      updatedTasks[i].status = "in_progress";
      setTasks([...updatedTasks]);
      setCurrentLog((prev) => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] 开始 ${updatedTasks[i].name}...`,
      ]);

      // Simulate progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        updatedTasks[i].progress = progress;
        setTasks([...updatedTasks]);
      }

      // Complete task
      updatedTasks[i].status = "completed";
      updatedTasks[i].progress = 100;
      setTasks([...updatedTasks]);
      setCurrentLog((prev) => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] ✓ ${updatedTasks[i].name} 完成`,
      ]);

      await new Promise((resolve) => setTimeout(resolve, 300));
    }

    setInstalling(false);
    updateWizardData({ installationComplete: true });
    setCurrentLog((prev) => [
      ...prev,
      `[${new Date().toLocaleTimeString()}] 🎉 所有安装任务已完成！`,
    ]);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "in_progress":
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case "error":
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return <div className="h-5 w-5 rounded-full border-2 border-muted" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <Badge variant="default" className="bg-green-500">完成</Badge>;
      case "in_progress":
        return <Badge variant="secondary">进行中</Badge>;
      case "error":
        return <Badge variant="destructive">错误</Badge>;
      default:
        return <Badge variant="outline">等待</Badge>;
    }
  };

  const allCompleted = tasks.every((t) => t.status === "completed");
  const overallProgress = Math.round(
    tasks.reduce((sum, task) => sum + task.progress, 0) / tasks.length
  );

  return (
    <WizardLayout
      title="安装依赖"
      description="下载并安装所需的运行环境和模型文件"
      hideNextButton={!allCompleted}
      onNext={() => {
        // Allow proceeding only when all tasks are completed
      }}
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
              {installing
                ? "正在安装，请稍候..."
                : allCompleted
                  ? "所有任务已完成！"
                  : "点击开始按钮开始安装"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Overall Progress */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-medium">总体进度</span>
                <span className="text-muted-foreground">{overallProgress}%</span>
              </div>
              <Progress value={overallProgress} className="h-2" />
            </div>

            {/* Installation Info */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">安装路径</p>
                <p className="font-mono font-medium truncate">
                  {wizardData.installPath}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">选择服务</p>
                <p className="font-medium">
                  {wizardData.selectedServices.length} 个服务
                </p>
              </div>
            </div>

            {/* Start Button */}
            {!installing && !allCompleted && (
              <button
                onClick={simulateInstallation}
                className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
              >
                <Download className="h-4 w-4" />
                开始安装
              </button>
            )}

            {allCompleted && (
              <Alert className="bg-green-50 border-green-200">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-800">
                  安装成功！您可以继续下一步启动服务。
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Task List */}
        <Card>
          <CardHeader>
            <CardTitle>安装任务</CardTitle>
            <CardDescription>各项安装任务的详细状态</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {tasks.map((task) => (
              <div
                key={task.id}
                className="flex items-center justify-between rounded-lg border p-4"
              >
                <div className="flex items-center gap-3 flex-1">
                  {getStatusIcon(task.status)}
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="font-medium">{task.name}</p>
                      {getStatusBadge(task.status)}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {task.description}
                    </p>
                    {task.status === "in_progress" && (
                      <div className="mt-2">
                        <Progress value={task.progress} className="h-1" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Installation Logs */}
        {currentLog.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>安装日志</CardTitle>
              <CardDescription>实时安装过程输出</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md bg-muted p-4 font-mono text-xs max-h-64 overflow-y-auto">
                {currentLog.map((log, idx) => (
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
