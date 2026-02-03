import { useContext } from "react";
import { WizardContext } from "@/context/wizardContext";

export function useWizard() {
  const context = useContext(WizardContext);
  if (context === undefined) {
    throw new Error("useWizard must be used within a WizardProvider");
  }
  return context;
}
