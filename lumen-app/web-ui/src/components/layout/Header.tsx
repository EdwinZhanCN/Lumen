import { Home, Moon, Sun } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useLumenSession } from "@/hooks/useLumenSession";
import { useTheme } from "@/hooks/useTheme";

export function Header() {
  const navigate = useNavigate();
  const { currentPath } = useLumenSession();
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-14 items-center px-4 lg:px-6">
        <div className="flex items-center gap-2">
          <div className="h-6 w-6 rounded-lg bg-primary" />
          <span className="font-semibold">Lumen</span>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate(currentPath ? "/session" : "/open")}
            title="回到会话入口"
          >
            <Home className="h-5 w-5" />
          </Button>
          <Button variant="ghost" size="icon" onClick={toggleTheme}>
            {theme === "dark" ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>
    </header>
  );
}
