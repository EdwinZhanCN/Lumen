export type Region = "cn" | "other";

export type Runtime = "onnx" | "rknn";

export interface WizardData {
  // Step 1: Welcome
  installPath: string;
  region: Region;
  serviceName: string;
  port: number;

  // Step 2: Hardware
  hardwarePreset: string | null;
  runtime: Runtime | null;
  onnxProviders: string[] | null;
  rknnDevice: string | null;

  // Step 3: Config
  servicePreset: "minimal" | "light_weight" | "basic" | "brave" | null;
  selectedServices: string[];
  configGenerated: boolean;
  configPath: string | undefined;

  // Step 4: Install
  installationComplete: boolean;

  // Step 5: Server
  serverRunning: boolean;
}

export type WizardStep = {
  id: "welcome" | "hardware" | "config" | "install" | "server";
  name: string;
  path: string;
};

export const WIZARD_STEPS: WizardStep[] = [
  { id: "welcome", name: "欢迎", path: "/welcome" },
  { id: "hardware", name: "硬件配置", path: "/hardware" },
  { id: "config", name: "服务配置", path: "/config" },
  { id: "install", name: "安装下载", path: "/install" },
  { id: "server", name: "启动服务", path: "/server" },
];
