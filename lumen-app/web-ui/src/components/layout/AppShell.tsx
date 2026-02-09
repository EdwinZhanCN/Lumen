import { Outlet } from "react-router-dom";
import { Header } from "@/components/layout/Header";

export function AppShell() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <Outlet />
    </div>
  );
}
