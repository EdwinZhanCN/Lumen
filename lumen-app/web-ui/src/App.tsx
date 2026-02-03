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
      <WizardProvider>
        <Routes>
          <Route path="/" element={<Navigate to="/welcome" replace />} />
          <Route path="/welcome" element={<Welcome />} />
          <Route path="/hardware" element={<Hardware />} />
          <Route path="/config" element={<Config />} />
          <Route path="/install" element={<Install />} />
          <Route path="/server" element={<Server />} />
        </Routes>
      </WizardProvider>
    </QueryClientProvider>
  );
}

export default App;
