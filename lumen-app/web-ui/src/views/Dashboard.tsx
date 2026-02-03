import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Cpu, Activity } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getServerStatus, getHardwareInfo, type ServerStatus, type HardwareInfo } from "@/lib/api";

export function Dashboard() {
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  async function loadData() {
    try {
      const [status, hwInfo] = await Promise.all([
        getServerStatus(),
        getHardwareInfo(),
      ]);
      setServerStatus(status);
      setHardwareInfo(hwInfo);
    } catch (error) {
      console.error("Failed to load dashboard data:", error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            Overview of your Lumen AI services
          </p>
        </div>
        <div className="flex gap-2">
          <Link to="/hardware">
            <Button variant="outline">
              <Cpu className="mr-2 h-4 w-4" />
              Hardware
            </Button>
          </Link>
          <Link to="/server">
            <Button>
              <Activity className="mr-2 h-4 w-4" />
              Server
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Server Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {loading ? (
                <span className="text-sm text-muted-foreground">
                  Loading...
                </span>
              ) : serverStatus?.running ? (
                <>
                  <div className="h-2 w-2 rounded-full bg-green-500" />
                  <span className="text-sm font-medium">Running</span>
                </>
              ) : (
                <>
                  <div className="h-2 w-2 rounded-full bg-red-500" />
                  <span className="text-sm font-medium">Stopped</span>
                </>
              )}
            </div>
            {serverStatus?.uptime_seconds && (
              <p className="text-xs text-muted-foreground">
                Uptime: {Math.floor(serverStatus.uptime_seconds / 60)}m
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Hardware</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {loading ? (
              <span className="text-sm text-muted-foreground">Loading...</span>
            ) : hardwareInfo ? (
              <div className="space-y-1">
                <p className="text-sm">
                  <span className="font-medium">Platform:</span>{" "}
                  {hardwareInfo.platform}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Recommended:</span>{" "}
                  {hardwareInfo.recommended_preset || "cpu"}
                </p>
              </div>
            ) : (
              <span className="text-sm text-muted-foreground">No data</span>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">-</div>
            <p className="text-xs text-muted-foreground">Configure services</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Installation</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">-</div>
            <p className="text-xs text-muted-foreground">Install drivers</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>
            Follow these steps to configure and run your Lumen AI services
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-start gap-4">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-medium">
              1
            </div>
            <div className="space-y-1">
              <h4 className="font-medium">Detect Hardware</h4>
              <p className="text-sm text-muted-foreground">
                Detect your hardware and select the best preset for your system.
              </p>
              <Link to="/hardware">
                <Button variant="link" className="h-auto p-0">
                  Go to Hardware →
                </Button>
              </Link>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-medium">
              2
            </div>
            <div className="space-y-1">
              <h4 className="font-medium">Configure Services</h4>
              <p className="text-sm text-muted-foreground">
                Generate a configuration file with your selected services.
              </p>
              <Link to="/config">
                <Button variant="link" className="h-auto p-0">
                  Go to Configuration →
                </Button>
              </Link>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-medium">
              3
            </div>
            <div className="space-y-1">
              <h4 className="font-medium">Install & Run</h4>
              <p className="text-sm text-muted-foreground">
                Install required drivers and start the Lumen server.
              </p>
              <div className="flex gap-2">
                <Link to="/install">
                  <Button variant="link" className="h-auto p-0">
                    Install →
                  </Button>
                </Link>
                <Link to="/server">
                  <Button variant="link" className="h-auto p-0">
                    Server →
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
