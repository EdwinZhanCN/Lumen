import { useEffect, useState } from "react";
import { FolderOpen, Info, Settings } from "lucide-react";
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
import { useLumenSession } from "@/hooks/useLumenSession";
import { useWizard } from "@/context/useWizard";
import type { Region } from "@/context/wizardConfig";

export function Welcome() {
  const { currentPath } = useLumenSession();
  const { wizardData, updateWizardData } = useWizard();

  const [region, setRegion] = useState<Region>(wizardData.region);
  const [serviceName, setServiceName] = useState(wizardData.serviceName);
  const [port, setPort] = useState(wizardData.port.toString());

  useEffect(() => {
    if (!currentPath) {
      return;
    }

    updateWizardData({
      installPath: currentPath,
      region,
      serviceName,
      port: parseInt(port, 10) || 50051,
    });
  }, [currentPath, port, region, serviceName, updateWizardData]);

  const handleConfigChange = (updates: {
    region?: Region;
    serviceName?: string;
    port?: string;
  }) => {
    if (updates.region !== undefined) {
      setRegion(updates.region);
    }
    if (updates.serviceName !== undefined) {
      setServiceName(updates.serviceName);
    }
    if (updates.port !== undefined) {
      setPort(updates.port);
    }

    updateWizardData({
      installPath: currentPath ?? "",
      region: updates.region ?? region,
      serviceName: updates.serviceName ?? serviceName,
      port: parseInt(updates.port ?? port, 10) || 50051,
    });
  };

  return (
    <WizardLayout title="基础配置" description="当前会话路径已固定，继续配置服务参数">
      <div className="space-y-6">
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            本次会话使用固定路径。若需修改路径，请点击右上角主页返回会话入口。
          </AlertDescription>
        </Alert>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="h-5 w-5" />
              会话路径
            </CardTitle>
            <CardDescription>该路径来自一级开屏验证结果</CardDescription>
          </CardHeader>
          <CardContent>
            <Input value={currentPath ?? ""} readOnly className="font-mono" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              服务器配置
            </CardTitle>
            <CardDescription>配置 gRPC 服务器基础参数</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
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
                    <SelectItem value="other">国际 (International)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="port">端口号 *</Label>
                <Input
                  id="port"
                  type="number"
                  min="1024"
                  max="65535"
                  value={port}
                  onChange={(e) => handleConfigChange({ port: e.target.value })}
                />
              </div>
            </div>

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
            </div>
          </CardContent>
        </Card>
      </div>
    </WizardLayout>
  );
}
