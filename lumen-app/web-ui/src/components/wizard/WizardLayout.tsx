import { type ReactNode } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWizard } from "@/context/useWizard";
import { WIZARD_STEPS } from "@/context/wizardConfig";
import { StepIndicator } from "./StepIndicator";

interface WizardLayoutProps {
  children: ReactNode;
  title: string;
  description?: string;
  onNext?: () => void | Promise<void>;
  onPrev?: () => void | Promise<void>;
  onFinish?: () => void | Promise<void>;
  hideNextButton?: boolean;
  hidePrevButton?: boolean;
  nextButtonText?: string;
  nextButtonDisabled?: boolean;
}

export function WizardLayout({
  children,
  title,
  description,
  onNext,
  onPrev,
  onFinish,
  hideNextButton = false,
  hidePrevButton = false,
  nextButtonText = "下一步",
  nextButtonDisabled = false,
}: WizardLayoutProps) {
  const { currentStep, nextStep, prevStep, canGoNext, canGoPrev } = useWizard();
  const isLastStep = currentStep === WIZARD_STEPS.length - 1;

  const handleNext = async () => {
    if (onNext) {
      await onNext();
    }

    if (isLastStep) {
      if (onFinish) {
        await onFinish();
      }
      return;
    }

    nextStep();
  };

  const handlePrev = async () => {
    if (onPrev) {
      await onPrev();
    }

    if (currentStep > 0) {
      prevStep();
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mx-auto max-w-4xl">
        <StepIndicator currentStep={currentStep} />

        <div className="mb-8">
          <h2 className="text-3xl font-bold">{title}</h2>
          {description && <p className="mt-2 text-muted-foreground">{description}</p>}
        </div>

        <div className="mb-8">{children}</div>

        <div className="flex items-center justify-between border-t pt-6">
          <div>
            {!hidePrevButton && canGoPrev() && (
              <Button variant="outline" onClick={handlePrev}>
                <ChevronLeft className="mr-2 h-4 w-4" />
                上一步
              </Button>
            )}
          </div>
          <div>
            {!hideNextButton && (
              <Button onClick={handleNext} disabled={!canGoNext() || nextButtonDisabled}>
                {isLastStep ? "完成" : nextButtonText}
                {!isLastStep && <ChevronRight className="ml-2 h-4 w-4" />}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
