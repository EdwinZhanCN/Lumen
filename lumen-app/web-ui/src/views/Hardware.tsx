import { useState, useEffect } from "react";
import {
  Cpu,
  Info,
  Laptop,
  Sparkles,
  Gamepad2,
  Rocket,
  Bot,
  Zap,
  Hexagon,
  Circle,
  Mountain,
  Loader2,
  AlertCircle,
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
  type HardwareInfo,
  type HardwarePreset,
} from "@/lib/api";

interface HardwarePresetWithIcon extends HardwarePreset {
  icon: React.ReactElement;
  recommended?: boolean;
}

// Icon mapping for different presets
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
  rockchip_rk3588: <Mountain />,
};

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

  // Fetch hardware info on mount
  useEffect(() => {
    const fetchHardwareInfo = async () => {
      try {
        setLoading(true);
        setError(null);
        const info = await getHardwareInfo();
        setHardwareInfo(info);

        // Transform API presets to UI presets with icons
        const presetsWithIcons: HardwarePresetWithIcon[] = (
          info.presets || []
        ).map((preset) => ({
          ...preset,
          icon: presetIcons[preset.name] || <Cpu />,
          recommended: preset.name === info.recommended_preset,
        }));

        setHardwarePresets(presetsWithIcons);

        // Auto-select recommended preset if none selected
        if (!selectedPreset && info.recommended_preset) {
          setSelectedPreset(info.recommended_preset);
        }
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

  useEffect(() => {
    if (selectedPreset) {
      const preset = hardwarePresets.find((p) => p.name === selectedPreset);
      if (preset) {
        updateWizardData({
          hardwarePreset: preset.name,
          runtime: preset.runtime as "onnx" | "rknn",
          onnxProviders:
            preset.providers && preset.providers.length > 0
              ? preset.providers
              : null,
          rknnDevice: preset.runtime === "rknn" ? "rk3588" : null,
        });
      }
    }
  }, [selectedPreset, hardwarePresets]);

  const handleSelectPreset = (presetName: string) => {
    setSelectedPreset(presetName);
  };

  const recommendedPresets = hardwarePresets.filter((p) => p.recommended);
  const otherPresets = hardwarePresets.filter((p) => !p.recommended);

  // Loading state
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

  // Error state
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
        {/* System Info Card */}
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
                  <p className="font-medium">
                    {hardwareInfo.processor || "N/A"}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Python版本</p>
                  <p className="font-medium">{hardwareInfo.python_version}</p>
                </div>
              </div>
              {hardwareInfo.recommended_preset && (
                <div className="mt-4 pt-4 border-t">
                  <div className="flex items-center gap-2">
                    <Badge variant="default" className="bg-green-500">
                      推荐配置
                    </Badge>
                    <span className="text-sm font-medium">
                      {hardwarePresets.find(
                        (p) => p.name === hardwareInfo.recommended_preset,
                      )?.name || hardwareInfo.recommended_preset}
                    </span>
                    {hardwareInfo.all_drivers_available && (
                      <Badge
                        variant="outline"
                        className="text-xs text-green-600 border-green-600"
                      >
                        ✓ 驱动就绪
                      </Badge>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Info Alert */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            选择与您的硬件匹配的配置预设。推荐配置已针对性能和兼容性进行优化。
          </AlertDescription>
        </Alert>

        {/* Recommended Presets */}
        {recommendedPresets.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold">推荐配置</h3>
              <Badge variant="default">优化推荐</Badge>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {recommendedPresets.map((preset) => (
                <Card
                  key={preset.name}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedPreset === preset.name
                      ? "border-primary bg-primary/5 shadow-md"
                      : "hover:border-primary/50"
                  }`}
                  onClick={() => handleSelectPreset(preset.name)}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="text-4xl mb-2">{preset.icon}</div>
                      {selectedPreset === preset.name && (
                        <Badge variant="default" className="bg-primary">
                          已选择
                        </Badge>
                      )}
                    </div>
                    <CardTitle className="text-lg">{preset.name}</CardTitle>
                    <CardDescription className="text-sm">
                      {preset.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-xs">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {preset.runtime.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="text-muted-foreground">
                        <p className="font-medium mb-1">提供商:</p>
                        <ul className="space-y-0.5">
                          {preset.providers && preset.providers.length > 0 ? (
                            preset.providers.map(
                              (provider: string, idx: number) => (
                                <li key={idx}>• {provider}</li>
                              ),
                            )
                          ) : (
                            <li>• 默认配置</li>
                          )}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Other Presets */}
        {otherPresets.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">其他配置</h3>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {otherPresets.map((preset) => (
                <Card
                  key={preset.name}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedPreset === preset.name
                      ? "border-primary bg-primary/5 shadow-md"
                      : "hover:border-primary/50"
                  }`}
                  onClick={() => handleSelectPreset(preset.name)}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="text-4xl mb-2">{preset.icon}</div>
                      {selectedPreset === preset.name && (
                        <Badge variant="default" className="bg-primary">
                          已选择
                        </Badge>
                      )}
                    </div>
                    <CardTitle className="text-lg">{preset.name}</CardTitle>
                    <CardDescription className="text-sm">
                      {preset.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-xs">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {preset.runtime.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="text-muted-foreground">
                        <p className="font-medium mb-1">提供商:</p>
                        <ul className="space-y-0.5">
                          {preset.providers && preset.providers.length > 0 ? (
                            preset.providers.map(
                              (provider: string, idx: number) => (
                                <li key={idx}>• {provider}</li>
                              ),
                            )
                          ) : (
                            <li>• 默认配置</li>
                          )}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Selected Preset Summary */}
        {selectedPreset && (
          <Card className="bg-muted/50">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                当前选择
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <div className="text-3xl">
                  {hardwarePresets.find((p) => p.name === selectedPreset)?.icon}
                </div>
                <div className="flex-1">
                  <p className="font-semibold">
                    {
                      hardwarePresets.find((p) => p.name === selectedPreset)
                        ?.name
                    }
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {
                      hardwarePresets.find((p) => p.name === selectedPreset)
                        ?.description
                    }
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </WizardLayout>
  );
}
