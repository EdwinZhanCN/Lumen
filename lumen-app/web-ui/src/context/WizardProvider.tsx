import {
  useState,
  useMemo,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { WIZARD_STEPS, type WizardData } from "./wizardConfig";
import { WizardContext } from "./wizardContext";
import { isValidPort, isValidServiceName } from "@/lib/wizardValidation";

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
  configGenerated: false,
  configPath: undefined,
  installationComplete: false,
  serverRunning: false,
};

export function WizardProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [wizardData, setWizardData] = useState<WizardData>(initialWizardData);

  // Compute currentStep from current route
  const currentStep = useMemo(() => {
    const stepIndex = WIZARD_STEPS.findIndex(
      (step) => step.path === location.pathname,
    );
    return stepIndex !== -1 ? stepIndex : 0;
  }, [location.pathname]);

  const updateWizardData = useCallback((data: Partial<WizardData>) => {
    setWizardData((prev) => ({ ...prev, ...data }));
  }, []);

  const isWelcomeComplete = useCallback((data: WizardData): boolean => {
    return (
      data.installPath.trim() !== "" &&
      isValidServiceName(data.serviceName) &&
      isValidPort(data.port)
    );
  }, []);

  const isHardwareComplete = useCallback((data: WizardData): boolean => {
    return data.hardwarePreset !== null;
  }, []);

  const isConfigComplete = useCallback((data: WizardData): boolean => {
    return data.selectedServices.length > 0 && data.configGenerated;
  }, []);

  const isInstallComplete = useCallback((data: WizardData): boolean => {
    return data.installationComplete;
  }, []);

  const getFirstBlockedStep = useCallback((): number => {
    if (!isWelcomeComplete(wizardData)) {
      return 0;
    }
    if (!isHardwareComplete(wizardData)) {
      return 1;
    }
    if (!isConfigComplete(wizardData)) {
      return 2;
    }
    if (!isInstallComplete(wizardData)) {
      return 3;
    }
    return WIZARD_STEPS.length - 1;
  }, [
    isConfigComplete,
    isHardwareComplete,
    isInstallComplete,
    isWelcomeComplete,
    wizardData,
  ]);

  // Navigation functions using React Router
  const goToStep = useCallback(
    (step: number) => {
      if (step < 0 || step >= WIZARD_STEPS.length) {
        return;
      }

      const firstBlockedStep = getFirstBlockedStep();
      if (step > firstBlockedStep) {
        navigate(WIZARD_STEPS[firstBlockedStep].path, { replace: true });
        return;
      }

      navigate(WIZARD_STEPS[step].path);
    },
    [getFirstBlockedStep, navigate],
  );

  const nextStep = useCallback(() => {
    if (currentStep < WIZARD_STEPS.length - 1) {
      navigate(WIZARD_STEPS[currentStep + 1].path);
    }
  }, [currentStep, navigate]);

  const prevStep = useCallback(() => {
    if (currentStep > 0) {
      navigate(WIZARD_STEPS[currentStep - 1].path);
    }
  }, [currentStep, navigate]);

  const canGoNext = useCallback((): boolean => {
    switch (currentStep) {
      case 0: // Welcome
        return isWelcomeComplete(wizardData);
      case 1: // Hardware
        return isHardwareComplete(wizardData);
      case 2: // Config
        return isConfigComplete(wizardData);
      case 3: // Install
        return isInstallComplete(wizardData);
      case 4: // Server
        return true;
      default:
        return true;
    }
  }, [
    currentStep,
    isConfigComplete,
    isHardwareComplete,
    isInstallComplete,
    isWelcomeComplete,
    wizardData,
  ]);

  const canGoPrev = useCallback((): boolean => {
    return currentStep > 0;
  }, [currentStep]);

  useEffect(() => {
    if (!location.pathname.startsWith("/setup/")) {
      return;
    }

    const targetStep = WIZARD_STEPS.findIndex(
      (step) => step.path === location.pathname,
    );
    if (targetStep < 0) {
      return;
    }

    const firstBlockedStep = getFirstBlockedStep();
    if (targetStep > firstBlockedStep) {
      navigate(WIZARD_STEPS[firstBlockedStep].path, { replace: true });
    }
  }, [getFirstBlockedStep, location.pathname, navigate]);

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
