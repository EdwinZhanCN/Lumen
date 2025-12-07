import { useMemo, useState } from "react";
import type { LumenConfig } from "../../types/lumen-config";
import { configToYaml } from "../../utils/yaml";

interface YamlPreviewProps {
  config: Partial<LumenConfig>;
  className?: string;
}

export function YamlPreview({ config, className = "" }: YamlPreviewProps) {
  const yamlContent = useMemo(() => configToYaml(config), [config]);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(yamlContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <div className={`card bg-base-200 shadow-xl ${className}`}>
      <div className="card-body">
        <div className="flex justify-between items-center mb-4">
          <h2 className="card-title text-lg">Configuration Preview</h2>
          <div className="flex gap-2">
            <button
              className={`btn btn-sm ${copied ? "btn-success" : "btn-ghost"}`}
              onClick={handleCopy}
              title="Copy to clipboard"
            >
              {copied ? (
                <>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                  Copied!
                </>
              ) : (
                <>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  Copy
                </>
              )}
            </button>
          </div>
        </div>

        <div className="mockup-code bg-base-300 text-sm overflow-auto max-h-[600px]">
          <pre className="px-6 py-4">
            <code className="language-yaml">{yamlContent}</code>
          </pre>
        </div>

        <div className="text-sm text-base-content/70 mt-2">
          <p>
            💡 This configuration will be saved to{" "}
            <code className="text-primary">~/.lumen/lumen.yaml</code>
          </p>
        </div>
      </div>
    </div>
  );
}
