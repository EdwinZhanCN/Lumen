import { useCallback, useState } from "react";

const STORAGE_KEY = "lumen.current_path";

function readStoredPath(): string | null {
  const value = localStorage.getItem(STORAGE_KEY);
  return value && value.trim() ? value : null;
}

export function useLumenSession() {
  const [currentPath, setCurrentPathState] = useState<string | null>(() =>
    readStoredPath(),
  );

  const setCurrentPath = useCallback((path: string) => {
    const normalized = path.trim();
    if (!normalized) {
      localStorage.removeItem(STORAGE_KEY);
      setCurrentPathState(null);
      return;
    }

    localStorage.setItem(STORAGE_KEY, normalized);
    setCurrentPathState(normalized);
  }, []);

  const clearCurrentPath = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setCurrentPathState(null);
  }, []);

  return {
    currentPath,
    setCurrentPath,
    clearCurrentPath,
  };
}
