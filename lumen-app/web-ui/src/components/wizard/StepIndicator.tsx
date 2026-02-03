import { Check } from "lucide-react";
import { WIZARD_STEPS } from "@/context/wizardConfig";

interface StepIndicatorProps {
  currentStep: number;
}

export function StepIndicator({ currentStep }: StepIndicatorProps) {
  return (
    <div className="w-full py-8">
      <div className="flex items-center justify-between max-w-4xl mx-auto">
        {WIZARD_STEPS.map((step, index) => (
          <div key={step.id} className="flex items-center flex-1">
            {/* Step Circle */}
            <div className="flex flex-col items-center">
              <div
                className={`flex h-10 w-10 items-center justify-center rounded-full border-2 font-semibold transition-colors ${
                  index < currentStep
                    ? "bg-primary border-primary text-primary-foreground"
                    : index === currentStep
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-muted bg-background text-muted-foreground"
                }`}
              >
                {index < currentStep ? (
                  <Check className="h-5 w-5" />
                ) : (
                  <span>{index + 1}</span>
                )}
              </div>
              <span
                className={`mt-2 text-xs font-medium ${
                  index <= currentStep
                    ? "text-foreground"
                    : "text-muted-foreground"
                }`}
              >
                {step.name}
              </span>
            </div>

            {/* Connector Line */}
            {index < WIZARD_STEPS.length - 1 && (
              <div className="flex-1 mx-2">
                <div
                  className={`h-0.5 transition-colors ${
                    index < currentStep ? "bg-primary" : "bg-muted"
                  }`}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
