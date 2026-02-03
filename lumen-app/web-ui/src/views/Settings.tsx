import { useState } from "react";
import { Settings as SettingsIcon, Save } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function Settings() {
  const [settings, setSettings] = useState({
    cacheDir: "~/.lumen",
    port: "8000",
    logLevel: "info",
  });

  const handleSave = () => {
    // TODO: Save settings to backend
    console.log("Saving settings:", settings);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground">
          Configure Lumen application settings
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <SettingsIcon className="h-5 w-5" />
            General Settings
          </CardTitle>
          <CardDescription>Basic application configuration</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="cacheDir">Cache Directory</Label>
            <Input
              id="cacheDir"
              value={settings.cacheDir}
              onChange={(e) =>
                setSettings({ ...settings, cacheDir: e.target.value })
              }
            />
            <p className="text-xs text-muted-foreground">
              Where downloaded models and data will be stored
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="port">Server Port</Label>
            <Input
              id="port"
              type="number"
              value={settings.port}
              onChange={(e) =>
                setSettings({ ...settings, port: e.target.value })
              }
            />
            <p className="text-xs text-muted-foreground">
              The port the backend server will run on
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="logLevel">Log Level</Label>
            <select
              id="logLevel"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              value={settings.logLevel}
              onChange={(e) =>
                setSettings({ ...settings, logLevel: e.target.value })
              }
            >
              <option value="debug">Debug</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
            </select>
            <p className="text-xs text-muted-foreground">
              The verbosity of logging output
            </p>
          </div>

          <Button onClick={handleSave}>
            <Save className="mr-2 h-4 w-4" />
            Save Settings
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
