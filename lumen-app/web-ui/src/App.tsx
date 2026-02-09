import type { ReactNode } from "react";
import { Navigate, Outlet, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { WizardProvider } from "@/context/WizardProvider";
import { useLumenSession } from "@/hooks/useLumenSession";
import { Config } from "@/views/Config";
import { Hardware } from "@/views/Hardware";
import { Install } from "@/views/Install";
import { OpenPath } from "@/views/OpenPath";
import { Server } from "@/views/Server";
import { SessionHub } from "@/views/SessionHub";
import { Welcome } from "@/views/Welcome";

const queryClient = new QueryClient();

function RequireSessionPath({ children }: { children: ReactNode }) {
  const { currentPath } = useLumenSession();

  if (!currentPath) {
    return <Navigate to="/open" replace />;
  }

  return <>{children}</>;
}

function SetupRoutes() {
  return (
    <RequireSessionPath>
      <WizardProvider>
        <Outlet />
      </WizardProvider>
    </RequireSessionPath>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Navigate to="/open" replace />} />
          <Route path="/open" element={<OpenPath />} />
          <Route
            path="/session"
            element={
              <RequireSessionPath>
                <SessionHub />
              </RequireSessionPath>
            }
          />
          <Route
            path="/server"
            element={
              <RequireSessionPath>
                <Server />
              </RequireSessionPath>
            }
          />

          <Route path="/setup" element={<SetupRoutes />}>
            <Route index element={<Navigate to="welcome" replace />} />
            <Route path="welcome" element={<Welcome />} />
            <Route path="hardware" element={<Hardware />} />
            <Route path="config" element={<Config />} />
            <Route path="install" element={<Install />} />
          </Route>
        </Route>

        <Route path="*" element={<Navigate to="/open" replace />} />
      </Routes>
    </QueryClientProvider>
  );
}

export default App;
