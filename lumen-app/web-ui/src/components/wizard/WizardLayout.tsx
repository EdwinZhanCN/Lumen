import { type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StepIndicator } from "./StepIndicator";
import { useWizard } from "@/context/useWizard";
import { WIZARD_STEPS } from "@/context/wizardConfig";

interface WizardLayoutProps {
  children: ReactNode;
  title: string;
  description?: string;
  onNext?: () => void | Promise<void>;
  onPrev?: () => void | Promise<void>;
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
  hideNextButton = false,
  hidePrevButton = false,
  nextButtonText = "下一步",
  nextButtonDisabled = false,
}: WizardLayoutProps) {
  const { currentStep, nextStep, prevStep, canGoNext, canGoPrev } = useWizard();
  const navigate = useNavigate();

  const handleNext = async () => {
    if (onNext) {
      await onNext();
    }
    if (currentStep < WIZARD_STEPS.length - 1) {
      nextStep();
      navigate(WIZARD_STEPS[currentStep + 1].path);
    }
  };

  const handlePrev = async () => {
    if (onPrev) {
      await onPrev();
    }
    if (currentStep > 0) {
      prevStep();
      navigate(WIZARD_STEPS[currentStep - 1].path);
    }
  };

  const isLastStep = currentStep === WIZARD_STEPS.length - 1;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-card">
        <div className="container mx-auto px-4">
          <div className="py-6">
            <h1 className="text-2xl font-bold">Lumen 配置向导</h1>
            <p className="text-sm text-muted-foreground mt-1">
              分布式推理系统配置工具
            </p>
          </div>
          <StepIndicator currentStep={currentStep} />
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Page Title */}
          <div className="mb-8">
            <h2 className="text-3xl font-bold">{title}</h2>
            {description && (
              <p className="text-muted-foreground mt-2">{description}</p>
            )}
          </div>

          {/* Page Content */}
          <div className="mb-8">{children}</div>

          {/* Navigation Buttons */}
          <div className="flex justify-between items-center pt-6 border-t">
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
                <Button
                  onClick={handleNext}
                  disabled={!canGoNext() || nextButtonDisabled}
                >
                  {isLastStep ? "完成" : nextButtonText}
                  {!isLastStep && <ChevronRight className="ml-2 h-4 w-4" />}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
