import { useState, useMemo, useCallback, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { WIZARD_STEPS, type WizardData } from "./wizardConfig";
import { WizardContext } from "./wizardContext";

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

  // Navigation functions using React Router
  const goToStep = useCallback(
    (step: number) => {
      if (step >= 0 && step < WIZARD_STEPS.length) {
        navigate(WIZARD_STEPS[step].path);
      }
    },
    [navigate],
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
        return wizardData.installPath.trim() !== "";
      case 1: // Hardware
        return wizardData.hardwarePreset !== null;
      case 2: // Config
        return (
          wizardData.selectedServices.length > 0 && wizardData.configGenerated
        );
      case 3: // Install
        return wizardData.installationComplete;
      case 4: // Server
        return true;
      default:
        return true;
    }
  }, [currentStep, wizardData]);

  const canGoPrev = useCallback((): boolean => {
    return currentStep > 0;
  }, [currentStep]);

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
