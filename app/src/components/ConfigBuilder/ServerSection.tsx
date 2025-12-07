import type { LumenConfig } from "../../types/lumen-config";

interface ServerSectionProps {
  server: LumenConfig["server"];
  onChange: (server: LumenConfig["server"]) => void;
}

export function ServerSection({ server, onChange }: ServerSectionProps) {
  const handlePortChange = (port: number) => {
    onChange({
      ...server,
      port,
    });
  };

  const handleHostChange = (host: string) => {
    onChange({
      ...server,
      host,
    });
  };

  const handleMeshEnabledChange = (enabled: boolean) => {
    onChange({
      ...server,
      mdns: {
        enabled,
        service_name: "lumen-service",
      },
    });
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
              d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01"
            />
          </svg>
          Server Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="form-control w-full">
            <label className="label pb-2">
              <span className="label-text font-medium flex items-center gap-2">
                Host Address
                <div
                  className="tooltip"
                  data-tip="0.0.0.0: all interfaces (remote access)&#10;127.0.0.1: localhost only (more secure)"
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
              placeholder="0.0.0.0"
              className="input input-bordered w-full font-mono"
              value={server.host || "0.0.0.0"}
              onChange={(e) => handleHostChange(e.target.value)}
            />
          </div>

          <div className="form-control w-full">
            <label className="label pb-2">
              <span className="label-text font-medium flex items-center gap-2">
                gRPC Port
                <div
                  className="tooltip"
                  data-tip="Common ports: 50051 (face), 50052 (clip), 50053 (ocr)&#10;Auto-assigns if port is occupied"
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
              type="number"
              placeholder="50051"
              className="input input-bordered w-full font-mono"
              value={server.port}
              onChange={(e) =>
                handlePortChange(parseInt(e.target.value) || 50051)
              }
              min={1024}
              max={65535}
            />
          </div>
        </div>

        <div className="divider">Lumen AI Mesh</div>

        <div className="form-control">
          <label className="label cursor-pointer justify-start gap-4">
            <input
              type="checkbox"
              className="checkbox checkbox-primary"
              checked={server.mdns?.enabled ?? false}
              onChange={(e) => handleMeshEnabledChange(e.target.checked)}
            />
            <div className="flex items-center gap-2">
              <span className="label-text font-medium">
                Enable Lumen AI Mesh
              </span>
              <div
                className="tooltip"
                data-tip="Enables automatic service discovery across your local network. Services can find and communicate with each other seamlessly."
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

        {server.mdns?.enabled && (
          <div className="alert alert-success ml-8 mt-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="stroke-current shrink-0 h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className="text-sm">
              Service will be discoverable as{" "}
              <code className="font-mono bg-base-300 px-1 py-0.5 rounded">
                lumen-service
              </code>{" "}
              on your network
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
