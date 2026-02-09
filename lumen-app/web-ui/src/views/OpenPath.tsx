import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, CheckCircle2, FolderOpen, Loader2, AlertCircle } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { validatePath, type PathValidationResponse } from "@/lib/api";
import { useLumenSession } from "@/hooks/useLumenSession";

type ValidationState =
  | { status: "idle" }
  | { status: "validating" }
  | { status: "validated"; result: PathValidationResponse }
  | { status: "error"; message: string };

export function OpenPath() {
  const navigate = useNavigate();
  const { currentPath, setCurrentPath } = useLumenSession();
  const [path, setPath] = useState(currentPath ?? "");
  const [validationState, setValidationState] = useState<ValidationState>({
    status: "idle",
  });

  const validatePathMutation = useMutation({ mutationFn: validatePath });

  const handleValidate = async () => {
    if (!path.trim()) {
      setValidationState({ status: "error", message: "请输入安装路径" });
      return;
    }

    setValidationState({ status: "validating" });

    try {
      const result = await validatePathMutation.mutateAsync({ path });
      setValidationState({ status: "validated", result });

      if (!result.valid) {
        return;
      }

      setCurrentPath(path);
      navigate("/session");
    } catch (error) {
      setValidationState({
        status: "error",
        message: error instanceof Error ? error.message : "路径验证失败",
      });
    }
  };

  const isValidating = validationState.status === "validating";
  const isValidated = validationState.status === "validated";

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mx-auto max-w-3xl space-y-6">
        <div>
          <h1 className="text-2xl font-semibold">Lumen 会话路径</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            输入本次会话的安装路径，验证成功后进入二级开屏。
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="h-5 w-5" />
              路径输入
            </CardTitle>
            <CardDescription>
              建议使用绝对路径，例如 ~/.lumen 或 /opt/lumen
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="path">安装路径 *</Label>
              <div className="flex gap-2">
                <Input
                  id="path"
                  value={path}
                  onChange={(e) => {
                    setPath(e.target.value);
                    if (validationState.status !== "idle") {
                      setValidationState({ status: "idle" });
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleValidate();
                    }
                  }}
                  placeholder="~/.lumen 或 /opt/lumen"
                  className="font-mono"
                  disabled={isValidating}
                />
                <Button
                  onClick={handleValidate}
                  disabled={isValidating || !path.trim()}
                >
                  {isValidating ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ArrowRight className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            {validationState.status === "error" && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{validationState.message}</AlertDescription>
              </Alert>
            )}

            {isValidated && validationState.result.error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{validationState.result.error}</AlertDescription>
              </Alert>
            )}

            {isValidated && validationState.result.valid && (
              <Alert className="border-green-500 bg-green-50">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-800">
                  路径有效 • 可用空间: {validationState.result.free_space_gb?.toFixed(1)} GB
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
