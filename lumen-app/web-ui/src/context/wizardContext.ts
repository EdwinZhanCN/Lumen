import { createContext } from "react";
import type { WizardData } from "@/context/wizardConfig";

export interface WizardContextType {
  currentStep: number;
  wizardData: WizardData;
  updateWizardData: (data: Partial<WizardData>) => void;
  goToStep: (step: number) => void;
  nextStep: () => void;
  prevStep: () => void;
  canGoNext: () => boolean;
  canGoPrev: () => boolean;
}

export const WizardContext = createContext<WizardContextType | undefined>(
  undefined,
);
