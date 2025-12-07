import type { LumenConfig } from "../../types/lumen-config";

interface DeploymentSectionProps {
  deployment: LumenConfig["deployment"];
  onChange: (deployment: LumenConfig["deployment"]) => void;
}

export function DeploymentSection({
  deployment,
  onChange,
}: DeploymentSectionProps) {
  const currentMode = (deployment as any).mode || "single";

  const handleModeChange = (mode: "single" | "hub") => {
    onChange({ mode } as LumenConfig["deployment"]);
  };

  return (
    <div className="card bg-base-100 shadow-lg">
      <div className="card-body">
        <h2 className="card-title text-xl mb-4">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
            />
          </svg>
          Deployment Mode
        </h2>

        <div className="form-control">
          <label className="label cursor-pointer justify-start gap-4">
            <input
              type="radio"
              name="deployment-mode"
              className="radio radio-primary"
              value="single"
              checked={currentMode === "single"}
              onChange={() => handleModeChange("single")}
            />
            <div className="flex-1 flex items-center gap-2">
              <span className="label-text font-medium">
                Single Mode (Recommended)
              </span>
              <div
                className="tooltip tooltip-right"
                data-tip="✓ Process isolation - one service crash won't affect others&#10;✓ Independent scaling - start/stop services individually&#10;⚠ Only one service per config"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 text-base-content/40 hover:text-base-content/70"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
            </div>
          </label>
        </div>

        <div className="divider my-2"></div>

        <div className="form-control">
          <label className="label cursor-pointer justify-start gap-4">
            <input
              type="radio"
              name="deployment-mode"
              className="radio radio-primary"
              value="hub"
              checked={currentMode === "hub"}
              onChange={() => handleModeChange("hub")}
            />
            <div className="flex-1 flex items-center gap-2">
              <span className="label-text font-medium">Hub Mode</span>
              <div className="badge badge-sm badge-warning">Coming Soon</div>
              <div
                className="tooltip tooltip-right"
                data-tip="✓ Lower memory footprint - shared Python runtime&#10;✓ Single port - all services via one gRPC endpoint&#10;✓ Multiple services in one config"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 text-base-content/40 hover:text-base-content/70"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
            </div>
          </label>
        </div>

        {currentMode === "hub" && (
          <div className="alert alert-info mt-4">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              className="stroke-current shrink-0 w-6 h-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="text-sm">
              <p>
                Hub mode is under development. Please use single mode for now.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
