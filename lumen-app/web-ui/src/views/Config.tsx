import { useState, useEffect, useRef, useCallback } from "react";
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
import { Button } from "@/components/ui/button";
import { WizardLayout } from "@/components/wizard/WizardLayout";
import { useWizard } from "@/context/useWizard";
import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { generateConfig } from "@/lib/api";
import { describeUiError } from "@/lib/errorPresentation";

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

type ConfigType = ServicePreset["id"];

const servicePresets: ServicePreset[] = [
  {
    id: "minimal",
    name: "极简",
    description:
      "仅包含图片文字识别服务，可以对文档/票据进行识别分类，可以搜索图片中的文字",
    icon: <StickyNote />,
    services: ["ocr"],
    recommended: false,
    requirements: "最小资源占用，适合轻量级部署",
  },
  {
    id: "light_weight",
    name: "轻量级",
    description: "提供照片文字识别，语义搜索，人脸识别，场景识别",
    icon: <Feather />,
    services: ["ocr", "clip", "face"],
    recommended: true,
    requirements: "平衡性能和资源，适合大多数场景",
  },
  {
    id: "basic",
    name: "基础版",
    description: "提供照片文字识别，语义搜索，人脸识别，场景识别，照片描述",
    icon: <Package />,
    services: ["ocr", "clip", "face", "vlm"],
    recommended: true,
    requirements: "完整功能，推荐用于体验高级生态",
  },
  {
    id: "brave",
    name: "激进版",
    description: "比基础版更加精确",
    icon: <Rocket />,
    services: ["ocr", "clip", "face", "vlm"],
    recommended: false,
    requirements: "高性能配置，需要强大硬件支持",
  },
];

const availableServices: Service[] = [
  {
    id: "ocr",
    name: "文字识别服务",
    description: "识别图片中的文字，用于分类和检索",
    package: "lumen-ocr",
    icon: <ScanText />,
    required: true,
  },
  {
    id: "clip",
    name: "图文理解服务",
    description: "提取图像和文本特征，用于分类、检索和相似度匹配",
    package: "lumen-clip",
    icon: <Image />,
    required: false,
  },
  {
    id: "face",
    name: "人脸识别服务",
    description: "检测图片中的人脸，对不同人物的照片进行分类",
    package: "lumen-face",
    icon: <ScanFace />,
    required: false,
  },
  {
    id: "vlm",
    name: "图片描述服务",
    description: "理解图片内容，并根据图片内容生成描述",
    package: "lumen-vlm",
    icon: <Bot />,
    required: false,
  },
];

export function Config() {
  const { wizardData, updateWizardData } = useWizard();
  const [selectedPreset, setSelectedPreset] = useState<ConfigType | null>(
    wizardData.servicePreset,
  );
  const lastConfigKeyRef = useRef<string | null>(null);

  // Use generated schema types with custom hook
  const {
    mutate: generateConfigMutate,
    reset: resetConfigMutation,
    data: configResult,
    isPending: generatingConfig,
    error: configError,
  } = useMutation({
    mutationFn: generateConfig,
    onSuccess: (data) => {
      if (!data.success) {
        updateWizardData({
          configGenerated: false,
          configPath: undefined,
        });
        return;
      }

      updateWizardData({
        configGenerated: true,
        configPath: data.config_path || undefined,
      });
    },
    onError: () => {
      updateWizardData({
        configGenerated: false,
        configPath: undefined,
      });
    },
  });
  const mutationErrorInfo = configError
    ? describeUiError(configError, "配置生成失败")
    : null;
  const businessErrorInfo =
    configResult && !configResult.success
      ? { title: "业务校验失败", message: configResult.message }
      : null;
  const errorInfo = mutationErrorInfo || businessErrorInfo;

  const requestConfigGeneration = useCallback(
    (presetId: ConfigType, force: boolean) => {
      const preset = servicePresets.find((item) => item.id === presetId);
      if (!preset || !wizardData.hardwarePreset || generatingConfig) {
        return;
      }

      const nextConfigKey = [
        presetId,
        wizardData.hardwarePreset,
        wizardData.installPath || "~/.lumen",
        wizardData.region,
        String(wizardData.port),
        wizardData.serviceName,
      ].join("|");

      if (!force && nextConfigKey === lastConfigKeyRef.current) {
        return;
      }

      lastConfigKeyRef.current = nextConfigKey;
      resetConfigMutation();
      updateWizardData({
        servicePreset: preset.id,
        selectedServices: preset.services,
        configGenerated: false,
        configPath: undefined,
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
    },
    [
      generatingConfig,
      generateConfigMutate,
      resetConfigMutation,
      updateWizardData,
      wizardData.hardwarePreset,
      wizardData.installPath,
      wizardData.port,
      wizardData.region,
      wizardData.serviceName,
    ],
  );

  // Auto-generate config when preset is selected
  useEffect(() => {
    if (!selectedPreset) {
      return;
    }

    requestConfigGeneration(selectedPreset, false);
  }, [selectedPreset, requestConfigGeneration]);

  const handleSelectPreset = (presetId: ConfigType) => {
    if (presetId === selectedPreset) {
      requestConfigGeneration(presetId, true);
      return;
    }

    setSelectedPreset(presetId);
  };

  const handleRetryGenerate = () => {
    if (!selectedPreset) {
      return;
    }
    requestConfigGeneration(selectedPreset, true);
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
        {errorInfo && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              <div className="space-y-3">
                <div className="space-y-1">
                  <p className="font-medium">{errorInfo.title}</p>
                  <p>{errorInfo.message}</p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleRetryGenerate}
                  disabled={!selectedPreset || generatingConfig}
                >
                  重试生成配置
                </Button>
              </div>
            </AlertDescription>
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
              <div className="grid gap-3 md:grid-cols-2">
                {wizardData.selectedServices.map((serviceId) => {
                  const service = availableServices.find(
                    (s) => s.id === serviceId,
                  );
                  if (!service) {
                    return null;
                  }
                  return (
                    <div
                      key={serviceId}
                      className="rounded-lg border bg-background px-3 py-3"
                    >
                      <div className="flex items-center gap-2 text-sm font-medium">
                        {service.icon}
                        <span>{service.name}</span>
                        {service.required && (
                          <Badge
                            variant="secondary"
                            className="ml-auto text-[11px]"
                          >
                            必选
                          </Badge>
                        )}
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">
                        {service.description}
                      </p>
                    </div>
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
