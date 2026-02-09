import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle, ArrowRight, Play, Settings } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { checkInstallationPath } from "@/lib/api";
import { useLumenSession } from "@/hooks/useLumenSession";

export function SessionHub() {
  const navigate = useNavigate();
  const { currentPath } = useLumenSession();

  const {
    data: sessionStatus,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["sessionStatus", currentPath],
    queryFn: () => checkInstallationPath(currentPath!),
    enabled: !!currentPath,
  });

  const canStartExisting =
    sessionStatus?.ready_to_start &&
    sessionStatus?.recommended_action === "start_existing";

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">会话入口</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              当前路径：<span className="font-mono">{currentPath}</span>
            </p>
          </div>
          <Button variant="outline" onClick={() => navigate("/open")}>切换路径</Button>
        </div>

        {isLoading && (
          <Card>
            <CardContent className="py-8 text-sm text-muted-foreground">正在检查路径状态...</CardContent>
          </Card>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              路径状态检查失败，请重新验证路径。
            </AlertDescription>
          </Alert>
        )}

        {!isLoading && !error && sessionStatus && (
          <>
            <Alert>
              <AlertDescription>{sessionStatus.message}</AlertDescription>
            </Alert>

            {canStartExisting ? (
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Play className="h-5 w-5" />
                      进入 Server 管理
                    </CardTitle>
                    <CardDescription>
                      使用当前已有环境和服务，直接启动或监控。
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button className="w-full" onClick={() => navigate("/server")}>进入 Server</Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Settings className="h-5 w-5" />
                      进入重新配置
                    </CardTitle>
                    <CardDescription>
                      保持当前路径，重新选择硬件、服务和安装流程。
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => navigate("/setup/welcome")}
                    >
                      进入配置
                    </Button>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <ArrowRight className="h-5 w-5" />
                    新的配置
                  </CardTitle>
                  <CardDescription>
                    当前路径没有可直接启动的服务，请先完成配置和安装。
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex items-center justify-between gap-4">
                  <Badge variant="outline">新配置</Badge>
                  <Button onClick={() => navigate("/setup/welcome")}>开始配置</Button>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  );
}
