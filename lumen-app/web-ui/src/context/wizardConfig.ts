export type Region = "cn" | "other";

export type Runtime = "onnx" | "rknn";

export interface WizardData {
  // Step 1: Welcome
  installPath: string;
  region: Region;
  serviceName: string;
  port: number;
  useExistingInstallation?: boolean;
  existingInstallationPath?: string;

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
  id: "welcome" | "hardware" | "config" | "install";
  name: string;
  path: string;
};

export const WIZARD_STEPS: WizardStep[] = [
  { id: "welcome", name: "基础配置", path: "/setup/welcome" },
  { id: "hardware", name: "硬件配置", path: "/setup/hardware" },
  { id: "config", name: "服务配置", path: "/setup/config" },
  { id: "install", name: "安装下载", path: "/setup/install" },
];
