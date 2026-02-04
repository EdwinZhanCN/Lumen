import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useDebounce } from "./useDebounce";
import { validatePath, checkInstallationPath } from "@/lib/api";
import type { PathValidationResponse, CheckInstallationPathResponse } from "@/lib/api";

export interface PathCheckState {
  path: string;
  setPath: (path: string) => void;
  validation: PathValidationResponse | undefined;
  installationStatus: CheckInstallationPathResponse | undefined;
  isLoading: boolean;
  isValidating: boolean;
  isCheckingInstallation: boolean;
}

/**
 * Hook to check installation path with debouncing.
 *
 * Features:
 * - Debounces path input (500ms)
 * - Validates path first
 * - If valid, checks for existing installation
 * - Returns loading states and results
 */
export function usePathCheck(debounceMs = 500): PathCheckState {
  const [path, setPath] = useState("");
  const debouncedPath = useDebounce(path, debounceMs);

  // Validate path first
  const {
    mutate: validatePathMutation,
    data: validation,
    isPending: isValidating,
  } = useMutation({
    mutationFn: validatePath,
  });

  // Then check installation status if path is valid
  const {
    data: installationStatus,
    isPending: isCheckingInstallation,
  } = useQuery({
    queryKey: ["checkInstallationPath", debouncedPath],
    queryFn: () => checkInstallationPath(debouncedPath),
    enabled: !!debouncedPath && validation?.valid === true,
    staleTime: 30000, // Cache for 30 seconds
  });

  // Trigger path validation when debounced path changes
  useState(() => {
    if (debouncedPath && debouncedPath.trim() !== "") {
      validatePathMutation({ path: debouncedPath });
    }
  });

  const isLoading = isValidating || isCheckingInstallation;

  return {
    path,
    setPath,
    validation,
    installationStatus,
    isLoading,
    isValidating,
    isCheckingInstallation,
  };
}
