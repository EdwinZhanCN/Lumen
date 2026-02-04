import { Routes, Route, Navigate } from "react-router-dom";
import { WizardProvider } from "./context/WizardProvider";
import { Welcome } from "@/views/Welcome";
import { Hardware } from "@/views/Hardware";
import { Config } from "@/views/Config";
import { Install } from "@/views/Install";
import { Server } from "@/views/Server";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Routes>
        {/* Wizard routes - wrapped in WizardProvider */}
        <Route
          path="/"
          element={
            <WizardProvider>
              <Navigate to="/welcome" replace />
            </WizardProvider>
          }
        />
        <Route
          path="/welcome"
          element={
            <WizardProvider>
              <Welcome />
            </WizardProvider>
          }
        />
        <Route
          path="/hardware"
          element={
            <WizardProvider>
              <Hardware />
            </WizardProvider>
          }
        />
        <Route
          path="/config"
          element={
            <WizardProvider>
              <Config />
            </WizardProvider>
          }
        />
        <Route
          path="/install"
          element={
            <WizardProvider>
              <Install />
            </WizardProvider>
          }
        />

        {/* Server page - independent, no WizardProvider */}
        <Route path="/server" element={<Server />} />
      </Routes>
    </QueryClientProvider>
  );
}

export default App;
