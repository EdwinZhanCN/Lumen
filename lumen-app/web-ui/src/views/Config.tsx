import { useState, useEffect, useRef } from "react";
import {
  Bot,
  Feather,
  Image,
  Info,
  Package,
  Rocket,
  ScanFace,
  ScanText,
  StickyNote,
  Loader2,
  CheckCircle2,
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
import { useMutation } from "@tanstack/react-query";
import { generateConfig } from "@/lib/api";

interface ServicePreset {
  id: "minimal" | "light_weight" | "basic" | "brave";
  name: string;
  description: string;
  icon: React.ReactElement;
  services: string[];
  recommended: boolean;
  requirements: string;
}

interface Service {
  id: string;
  name: string;
  description: string;
  package: string;
  icon: React.ReactElement;
  required: boolean;
}

const servicePresets: ServicePreset[] = [
  {
    id: "minimal",
    name: "最小化",
    description: "仅包含 OCR 服务",
    icon: <StickyNote />,
    services: ["ocr"],
    recommended: false,
    requirements: "最小资源占用，适合轻量级部署",
  },
  {
    id: "light_weight",
    name: "轻量级",
    description: "OCR + CLIP + Face",
    icon: <Feather />,
    services: ["ocr", "clip", "face"],
    recommended: true,
    requirements: "平衡性能和资源，适合大多数场景",
  },
  {
    id: "basic",
    name: "基础版",
    description: "OCR + CLIP + Face + VLM",
    icon: <Package />,
    services: ["ocr", "clip", "face", "vlm"],
    recommended: true,
    requirements: "完整功能，推荐用于生产环境",
  },
  {
    id: "brave",
    name: "完整版",
    description: "所有服务 + 高性能模型",
    icon: <Rocket />,
    services: ["ocr", "clip", "face", "vlm"],
    recommended: false,
    requirements: "高性能配置，需要强大硬件支持",
  },
];

const availableServices: Service[] = [
  {
    id: "ocr",
    name: "OCR",
    description: "文字识别服务 (PP-OCRv5)",
    package: "lumen-ocr",
    icon: <ScanText />,
    required: true,
  },
  {
    id: "clip",
    name: "CLIP",
    description: "视觉-语言理解 (MobileCLIP/CN-CLIP)",
    package: "lumen-clip",
    icon: <Image />,
    required: false,
  },
  {
    id: "face",
    name: "Face",
    description: "人脸检测与识别 (Buffalo/Antelope)",
    package: "lumen-face",
    icon: <ScanFace />,
    required: false,
  },
  {
    id: "vlm",
    name: "VLM",
    description: "视觉语言模型 (FastVLM-0.5B)",
    package: "lumen-vlm",
    icon: <Bot />,
    required: false,
  },
];

export function Config() {
  const { wizardData, updateWizardData } = useWizard();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(
    wizardData.servicePreset,
  );
  const lastConfigKeyRef = useRef<string | null>(null);

  // Use generated schema types with custom hook
  const {
    mutate: generateConfigMutate,
    data: configResult,
    isPending: generatingConfig,
    error: configError,
  } = useMutation({
    mutationFn: generateConfig,
    onSuccess: (data) => {
      updateWizardData({
        configGenerated: true,
        configPath: data.config_path || undefined,
      });
    },
    onError: () => {
      updateWizardData({ configGenerated: false });
    },
  });
  const error =
    configError?.message ||
    (configResult && !configResult.success ? configResult.message : null);

  // Auto-generate config when preset is selected
  useEffect(() => {
    if (!selectedPreset) {
      return;
    }

    const preset = servicePresets.find((p) => p.id === selectedPreset);
    if (!preset) {
      return;
    }

    if (!wizardData.hardwarePreset) {
      return;
    }

    const nextConfigKey = [
      selectedPreset,
      wizardData.hardwarePreset,
      wizardData.installPath || "~/.lumen",
      wizardData.region,
      String(wizardData.port),
      wizardData.serviceName,
    ].join("|");

    if (nextConfigKey === lastConfigKeyRef.current || generatingConfig) {
      return;
    }

    lastConfigKeyRef.current = nextConfigKey;

    updateWizardData({
      servicePreset: preset.id,
      selectedServices: preset.services,
      configGenerated: false,
    });

    generateConfigMutate({
      preset: wizardData.hardwarePreset,
      region: wizardData.region,
      cache_dir: wizardData.installPath || "~/.lumen",
      port: wizardData.port,
      service_name: wizardData.serviceName,
      config_type: preset.id,
      clip_model: null,
    });
  }, [
    selectedPreset,
    wizardData.hardwarePreset,
    wizardData.installPath,
    wizardData.region,
    wizardData.port,
    wizardData.serviceName,
    generateConfigMutate,
    updateWizardData,
    generatingConfig,
  ]);

  const handleSelectPreset = (presetId: string) => {
    setSelectedPreset(presetId);
  };

  const recommendedPresets = servicePresets.filter((p) => p.recommended);
  const otherPresets = servicePresets.filter((p) => !p.recommended);

  return (
    <WizardLayout title="配置服务" description="选择您需要启用的 AI 服务">
      <div className="space-y-6">
        {/* Info Alert */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            选择适合您需求的服务预设配置。配置将基于您选择的硬件预设（
            {wizardData.hardwarePreset || "未选择"}）自动生成。
          </AlertDescription>
        </Alert>

        {/* Config Generation Status */}
        {generatingConfig && (
          <Alert>
            <Loader2 className="h-4 w-4 animate-spin" />
            <AlertDescription>正在生成配置文件...</AlertDescription>
          </Alert>
        )}

        {/* Config Generation Success */}
        {configResult && configResult.success && (
          <Alert className="border-green-500 bg-green-50">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-800">
              配置生成成功！路径: {configResult.config_path}
            </AlertDescription>
          </Alert>
        )}

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="space-y-6">
          {/* Recommended Presets */}
          {recommendedPresets.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-semibold">推荐配置</h3>
                <Badge variant="default">优化推荐</Badge>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                {recommendedPresets.map((preset) => (
                  <Card
                    key={preset.id}
                    className={`cursor-pointer transition-all hover:shadow-md ${
                      selectedPreset === preset.id
                        ? "border-primary bg-primary/5 shadow-md"
                        : "hover:border-primary/50"
                    }`}
                    onClick={() => handleSelectPreset(preset.id)}
                  >
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="text-4xl mb-2">{preset.icon}</div>
                        {selectedPreset === preset.id && (
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
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-2">
                            包含服务:
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {preset.services.map((serviceId) => {
                              const service = availableServices.find(
                                (s) => s.id === serviceId,
                              );
                              return (
                                <Badge
                                  key={serviceId}
                                  variant="secondary"
                                  className="flex text-xs gap-2"
                                >
                                  {service?.icon} {service?.name}
                                </Badge>
                              );
                            })}
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {preset.requirements}
                        </p>
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
              <div className="grid gap-4 md:grid-cols-2">
                {otherPresets.map((preset) => (
                  <Card
                    key={preset.id}
                    className={`cursor-pointer transition-all hover:shadow-md ${
                      selectedPreset === preset.id
                        ? "border-primary bg-primary/5 shadow-md"
                        : "hover:border-primary/50"
                    }`}
                    onClick={() => handleSelectPreset(preset.id)}
                  >
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="text-4xl mb-2">{preset.icon}</div>
                        {selectedPreset === preset.id && (
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
                      <div className="space-y-3">
                        <div>
                          <p className="text-xs font-medium text-muted-foreground mb-2">
                            包含服务:
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {preset.services.map((serviceId) => {
                              const service = availableServices.find(
                                (s) => s.id === serviceId,
                              );
                              return (
                                <Badge
                                  key={serviceId}
                                  variant="secondary"
                                  className="flex text-xs gap-2"
                                >
                                  {service?.icon} {service?.name}
                                </Badge>
                              );
                            })}
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {preset.requirements}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Selected Services Summary */}
        <Card className="bg-muted/50">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Package className="h-4 w-4" />
              已选择的服务
            </CardTitle>
          </CardHeader>
          <CardContent>
            {wizardData.selectedServices.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {wizardData.selectedServices.map((serviceId) => {
                  const service = availableServices.find(
                    (s) => s.id === serviceId,
                  );
                  return (
                    <Badge
                      key={serviceId}
                      variant="secondary"
                      className="flex text-xs gap-2"
                    >
                      {service?.icon} {service?.name}
                    </Badge>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">尚未选择任何服务</p>
            )}
          </CardContent>
        </Card>
      </div>
    </WizardLayout>
  );
}
