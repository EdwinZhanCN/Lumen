import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import type { LumenConfig } from "../../types/lumen-config";
import { regionOptions } from "../../utils/defaultConfig";

interface MetadataSectionProps {
  metadata: LumenConfig["metadata"];
  onChange: (metadata: LumenConfig["metadata"]) => void;
}

export function MetadataSection({ metadata, onChange }: MetadataSectionProps) {
  const [isSelectingFolder, setIsSelectingFolder] = useState(false);

  const handleChange = (
    field: keyof LumenConfig["metadata"],
    value: string,
  ) => {
    onChange({
      ...metadata,
      [field]: value,
    });
  };

  const handleSelectFolder = async () => {
    try {
      setIsSelectingFolder(true);
      console.log("Opening folder picker...");

      const selected = await open({
        directory: true,
        multiple: false,
        title: "Select Cache Directory",
      });

      console.log("Selected folder:", selected);

      if (selected && typeof selected === "string") {
        handleChange("cache_dir", selected);
      } else if (selected === null) {
        console.log("Folder selection cancelled by user");
      }
    } catch (error) {
      console.error("Failed to select folder:", error);
      alert(`Failed to open folder picker: ${error}`);
    } finally {
      setIsSelectingFolder(false);
    }
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
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          Metadata
        </h2>

        <div className="form-control w-full">
          <label className="label pb-2">
            <span className="label-text font-medium flex items-center gap-2">
              Configuration Version
              <div
                className="tooltip"
                data-tip="Semantic versioning (MAJOR.MINOR.PATCH)"
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
            </span>
          </label>
          <input
            type="text"
            placeholder="1.0.0"
            className="input input-bordered w-full"
            value={metadata.version}
            onChange={(e) => handleChange("version", e.target.value)}
          />
        </div>

        <div className="form-control w-full">
          <label className="label pb-2">
            <span className="label-text font-medium flex items-center gap-2">
              Region
              <div
                className="tooltip"
                data-tip="International: HuggingFace&#10;China: ModelScope (faster in China)"
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
            </span>
          </label>
          <select
            className="select select-bordered w-full"
            value={metadata.region}
            onChange={(e) =>
              handleChange("region", e.target.value as "cn" | "other")
            }
          >
            {regionOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-control w-full">
          <label className="label pb-2">
            <span className="label-text font-medium flex items-center gap-2">
              Cache Directory
              <div
                className="tooltip"
                data-tip="Supports ~ expansion. Models stored in cache_dir/models. Config saved to cache_dir/lumen.yaml"
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
            </span>
          </label>
          <div className="join w-full">
            <input
              type="text"
              placeholder="~/.lumen"
              className="input input-bordered join-item flex-1 font-mono"
              value={metadata.cache_dir}
              onChange={(e) => handleChange("cache_dir", e.target.value)}
            />
            <button
              type="button"
              className="btn btn-primary join-item"
              onClick={handleSelectFolder}
              disabled={isSelectingFolder}
              title="Browse for folder"
            >
              {isSelectingFolder ? (
                <span className="loading loading-spinner loading-sm"></span>
              ) : (
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
                    d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
