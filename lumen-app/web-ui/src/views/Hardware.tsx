import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  Circle,
  Cpu,
  Gamepad2,
  Hexagon,
  Info,
  Laptop,
  Loader2,
  // Mountain,
  Rocket,
  Sparkles,
  Wrench,
  Zap,
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
import * as React from "react";
import {
  getHardwareInfo,
  type DriverStatus,
  type HardwareInfo,
  type HardwarePreset,
} from "@/lib/api";

interface HardwarePresetWithIcon extends HardwarePreset {
  icon: React.ReactElement;
  recommended?: boolean;
}

function isPresetSelectable(
  preset: Pick<HardwarePreset, "availability" | "ready" | "supported_on_current_platform"> | null | undefined,
): boolean {
  if (!preset || preset.supported_on_current_platform === false) {
    return false;
  }

  const availability = String(
    preset.availability ?? (preset.ready ? "ready" : "not_checked"),
  );
  return availability !== "incompatible";
}

const presetIcons: Record<string, React.ReactElement> = {
  cpu: <Cpu />,
  apple_silicon: <Laptop />,
  nvidia_gpu: <Gamepad2 />,
  nvidia_gpu_high: <Rocket />,
  nvidia_jetson: <Bot />,
  nvidia_jetson_high: <Sparkles />,
  intel_gpu: <Hexagon />,
  amd_gpu_win: <Circle />,
  amd_npu: <Zap />,
  // Rockchip support is temporarily disabled in lumen-app.
  // rockchip: <Mountain />,
};

function getAvailability(preset: HardwarePresetWithIcon): {
  label: string;
  className: string;
} {
  const availability = String(
    preset.availability ?? (preset.ready ? "ready" : "not_checked"),
  );
  if (availability === "ready") {
    return {
      label: "就绪",
      className: "text-green-700 border-green-600 bg-green-50",
    };
  }
  if (availability === "missing_drivers") {
    return {
      label: "缺少驱动",
      className: "text-amber-700 border-amber-600 bg-amber-50",
    };
  }
  if (availability === "incompatible") {
    return {
      label: "不兼容",
      className: "text-red-700 border-red-600 bg-red-50",
    };
  }
  return {
    label: "未检查",
    className: "text-muted-foreground border-muted",
  };
}

function getDriverStatusLabel(status: string): string {
  if (status === "available") {
    return "可用";
  }
  if (status === "incompatible") {
    return "不兼容";
  }
  return "缺失";
}

function getDriverStatusClassName(status: string): string {
  if (status === "available") {
    return "text-green-700 border-green-600 bg-green-50";
  }
  if (status === "incompatible") {
    return "text-red-700 border-red-600 bg-red-50";
  }
  return "text-amber-700 border-amber-600 bg-amber-50";
}

export function Hardware() {
  const { wizardData, updateWizardData } = useWizard();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(
    wizardData.hardwarePreset,
  );
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfo | null>(null);
  const [hardwarePresets, setHardwarePresets] = useState<
    HardwarePresetWithIcon[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHardwareInfo = async () => {
      try {
        setLoading(true);
        setError(null);
        const info = await getHardwareInfo();
        setHardwareInfo(info);

        const presetsWithIcons: HardwarePresetWithIcon[] = (
          info.presets || []
        ).map((preset) => ({
          ...preset,
          icon: presetIcons[preset.name] || <Cpu />,
          recommended: preset.name === info.recommended_preset,
        }));

        setHardwarePresets(presetsWithIcons);
        setSelectedPreset((current) => {
          if (current) {
            const selected = presetsWithIcons.find((preset) => preset.name === current);
            if (isPresetSelectable(selected)) {
              return current;
            }
          }
          if (info.recommended_preset) {
            const recommended = presetsWithIcons.find(
              (preset) => preset.name === info.recommended_preset,
            );
            if (isPresetSelectable(recommended)) {
              return info.recommended_preset;
            }
          }
          return (
            presetsWithIcons.find((preset) => isPresetSelectable(preset))?.name ?? null
          );
        });
      } catch (err) {
        console.error("Failed to fetch hardware info:", err);
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load hardware information",
        );
      } finally {
        setLoading(false);
      }
    };

    fetchHardwareInfo();
  }, []);

  const selectedPresetInfo = useMemo(
    () => hardwarePresets.find((preset) => preset.name === selectedPreset),
    [hardwarePresets, selectedPreset],
  );

  useEffect(() => {
    if (!selectedPresetInfo) {
      return;
    }
    if (!isPresetSelectable(selectedPresetInfo)) {
      return;
    }
    updateWizardData({
      hardwarePreset: selectedPresetInfo.name,
      runtime: selectedPresetInfo.runtime as "onnx" | "rknn",
      onnxProviders:
        selectedPresetInfo.providers && selectedPresetInfo.providers.length > 0
          ? selectedPresetInfo.providers
          : null,
      // Rockchip support is temporarily disabled in lumen-app.
      // rknnDevice: selectedPresetInfo.runtime === "rknn" ? "rk3588" : null,
      rknnDevice: null,
    });
  }, [selectedPresetInfo, updateWizardData]);

  const handleSelectPreset = (presetName: string) => {
    const preset = hardwarePresets.find((item) => item.name === presetName);
    if (!isPresetSelectable(preset)) {
      return;
    }
    setSelectedPreset(presetName);
  };

  const recommendedPresets = hardwarePresets.filter((preset) => preset.recommended);
  const otherPresets = hardwarePresets.filter((preset) => !preset.recommended);

  const renderPresetCard = (preset: HardwarePresetWithIcon) => {
    const availability = getAvailability(preset);
    const drivers = (preset.drivers || []) as DriverStatus[];
    const missingInstallable = preset.missing_installable || [];
    const isSelected = selectedPreset === preset.name;
    const isUnsupported = preset.supported_on_current_platform === false;
    const isUnavailable = !isPresetSelectable(preset);

    return (
      <Card
        key={preset.name}
        className={`transition-all ${
          isSelected
            ? "border-primary bg-primary/5 shadow-md"
            : isUnavailable
              ? ""
              : "hover:border-primary/50 hover:shadow-md"
        } ${isUnavailable ? "cursor-not-allowed opacity-70 border-dashed" : "cursor-pointer"}`}
        onClick={() => handleSelectPreset(preset.name)}
      >
        <CardHeader>
          <div className="flex items-start justify-between gap-2">
            <div className="text-4xl mb-2">{preset.icon}</div>
            <div className="flex gap-1">
              {isSelected && (
                <Badge variant="default" className="bg-primary">
                  已选择
                </Badge>
              )}
              <Badge variant="outline" className={availability.className}>
                {availability.label}
              </Badge>
            </div>
          </div>
          <CardTitle className="text-lg">{preset.name}</CardTitle>
          <CardDescription className="text-sm">{preset.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-xs">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {preset.runtime.toUpperCase()}
              </Badge>
              {isUnsupported && (
                <Badge variant="outline" className="text-xs border-red-600 text-red-700">
                  当前系统不可用
                </Badge>
              )}
              {!isUnsupported && isUnavailable && (
                <Badge variant="outline" className="text-xs border-red-600 text-red-700">
                  当前环境不兼容
                </Badge>
              )}
            </div>

            <div className="text-muted-foreground">
              <p className="font-medium mb-1">提供商:</p>
              <ul className="space-y-0.5">
                {preset.providers && preset.providers.length > 0 ? (
                  preset.providers.map((provider: string, idx: number) => (
                    <li key={idx}>• {provider}</li>
                  ))
                ) : (
                  <li>• 默认配置</li>
                )}
              </ul>
            </div>

            {preset.environment_checked && (
              <div className="space-y-1">
                <p className="font-medium text-muted-foreground">驱动状态:</p>
                {drivers.length > 0 ? (
                  drivers.map((driver, idx) => {
                    const status = String(driver.status || "missing");
                    return (
                      <div
                        key={`${driver.name}-${idx}`}
                        className="flex items-center justify-between gap-2"
                      >
                        <span className="text-muted-foreground">{driver.name}</span>
                        <Badge
                          variant="outline"
                          className={getDriverStatusClassName(status)}
                        >
                          {getDriverStatusLabel(status)}
                        </Badge>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-muted-foreground">无需额外驱动</div>
                )}
              </div>
            )}

            {missingInstallable.length > 0 && (
              <div className="rounded-md border border-amber-300 bg-amber-50 p-2 text-[11px] text-amber-700">
                可自动安装: {missingInstallable.join(", ")}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <WizardLayout
        title="选择硬件配置"
        description="根据您的硬件选择最适合的推理加速方案"
      >
        <div className="flex items-center justify-center py-12">
          <div className="text-center space-y-4">
            <Loader2 className="h-8 w-8 animate-spin mx-auto text-primary" />
            <p className="text-muted-foreground">正在检测硬件配置...</p>
          </div>
        </div>
      </WizardLayout>
    );
  }

  if (error) {
    return (
      <WizardLayout
        title="选择硬件配置"
        description="根据您的硬件选择最适合的推理加速方案"
      >
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </WizardLayout>
    );
  }

  return (
    <WizardLayout
      title="选择硬件配置"
      description="根据您的硬件选择最适合的推理加速方案"
    >
      <div className="space-y-6">
        {hardwareInfo && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Info className="h-4 w-4" />
                系统信息
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">平台</p>
                  <p className="font-medium">{hardwareInfo.platform}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">架构</p>
                  <p className="font-medium">{hardwareInfo.machine}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">处理器</p>
                  <p className="font-medium">{hardwareInfo.processor || "N/A"}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Python版本</p>
                  <p className="font-medium">{hardwareInfo.python_version}</p>
                </div>
              </div>
              {hardwareInfo.recommended_preset && (
                <div className="mt-4 pt-4 border-t">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="default" className="bg-green-600">
                      推荐配置
                    </Badge>
                    <span className="text-sm font-medium">
                      {hardwareInfo.recommended_preset}
                    </span>
                    {hardwareInfo.all_drivers_available ? (
                      <Badge
                        variant="outline"
                        className="text-xs text-green-700 border-green-600"
                      >
                        <CheckCircle2 className="h-3 w-3 mr-1" />
                        驱动就绪
                      </Badge>
                    ) : (
                      <Badge
                        variant="outline"
                        className="text-xs text-amber-700 border-amber-600"
                      >
                        <Wrench className="h-3 w-3 mr-1" />
                        需要处理驱动
                      </Badge>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            推荐项优先保证可用性。若选择“缺少驱动”的预设，安装阶段会尝试自动补齐可安装驱动。
          </AlertDescription>
        </Alert>

        {recommendedPresets.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold">推荐配置</h3>
              <Badge variant="default">优化推荐</Badge>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {recommendedPresets.map((preset) => renderPresetCard(preset))}
            </div>
          </div>
        )}

        {otherPresets.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">其他配置</h3>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {otherPresets.map((preset) => renderPresetCard(preset))}
            </div>
          </div>
        )}

        {selectedPresetInfo && (
          <Card className="bg-muted/50">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                当前选择
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-start gap-4">
                <div className="text-3xl">{selectedPresetInfo.icon}</div>
                <div className="flex-1 space-y-2">
                  <p className="font-semibold">{selectedPresetInfo.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {selectedPresetInfo.description}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline" className={getAvailability(selectedPresetInfo).className}>
                      {getAvailability(selectedPresetInfo).label}
                    </Badge>
                    {selectedPresetInfo.supported_on_current_platform === false && (
                      <Badge variant="outline" className="border-red-600 text-red-700">
                        当前平台不支持
                      </Badge>
                    )}
                  </div>
                  {(selectedPresetInfo.missing_installable || []).length > 0 && (
                    <div className="text-xs text-amber-700">
                      可自动安装驱动:{" "}
                      {(selectedPresetInfo.missing_installable || []).join(", ")}
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </WizardLayout>
  );
}
