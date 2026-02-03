import { createContext, useContext, useState, type ReactNode } from "react";

export type Region = "cn" | "international";

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
  onnxProviders: any[] | null;
  rknnDevice: string | null;

  // Step 3: Config
  servicePreset: "minimal" | "light_weight" | "basic" | "brave" | null;
  selectedServices: string[];

  // Step 4: Install
  installationComplete: boolean;

  // Step 5: Server
  serverRunning: boolean;
}

export const WIZARD_STEPS = [
  { id: "welcome", name: "欢迎", path: "/welcome" },
  { id: "hardware", name: "硬件配置", path: "/hardware" },
  { id: "config", name: "服务配置", path: "/config" },
  { id: "install", name: "安装下载", path: "/install" },
  { id: "server", name: "启动服务", path: "/server" },
] as const;

export type WizardStepId = (typeof WIZARD_STEPS)[number]["id"];

interface WizardContextType {
  currentStep: number;
  wizardData: WizardData;
  updateWizardData: (data: Partial<WizardData>) => void;
  goToStep: (step: number) => void;
  nextStep: () => void;
  prevStep: () => void;
  canGoNext: () => boolean;
  canGoPrev: () => boolean;
}

const WizardContext = createContext<WizardContextType | undefined>(undefined);

const initialWizardData: WizardData = {
  installPath: "",
  region: "cn",
  serviceName: "lumen-server",
  port: 50051,
  hardwarePreset: null,
  runtime: null,
  onnxProviders: null,
  rknnDevice: null,
  servicePreset: null,
  selectedServices: ["ocr"],
  installationComplete: false,
  serverRunning: false,
};

export function WizardProvider({ children }: { children: ReactNode }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [wizardData, setWizardData] = useState<WizardData>(initialWizardData);

  const updateWizardData = (data: Partial<WizardData>) => {
    setWizardData((prev) => ({ ...prev, ...data }));
  };

  const goToStep = (step: number) => {
    if (step >= 0 && step < WIZARD_STEPS.length) {
      setCurrentStep(step);
    }
  };

  const nextStep = () => {
    if (currentStep < WIZARD_STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const canGoNext = (): boolean => {
    switch (currentStep) {
      case 0: // Welcome
        return wizardData.installPath.trim() !== "";
      case 1: // Hardware
        return wizardData.hardwarePreset !== null;
      case 2: // Config
        return wizardData.selectedServices.length > 0;
      case 3: // Install
        return wizardData.installationComplete;
      case 4: // Server
        return true;
      default:
        return true;
    }
  };

  const canGoPrev = (): boolean => {
    return currentStep > 0;
  };

  return (
    <WizardContext.Provider
      value={{
        currentStep,
        wizardData,
        updateWizardData,
        goToStep,
        nextStep,
        prevStep,
        canGoNext,
        canGoPrev,
      }}
    >
      {children}
    </WizardContext.Provider>
  );
}

export function useWizard() {
  const context = useContext(WizardContext);
  if (context === undefined) {
    throw new Error("useWizard must be used within a WizardProvider");
  }
  return context;
}
