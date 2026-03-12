import { useEffect, useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useDebounce } from "./useDebounce";
import { validatePath, checkInstallationPath } from "@/lib/api";
import type {
  PathValidationResponse,
  CheckInstallationPathResponse,
} from "@/lib/api";

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
  const [validation, setValidation] = useState<PathValidationResponse>();
  const [validatedPath, setValidatedPath] = useState("");
  const debouncedPath = useDebounce(path, debounceMs);

  const { mutateAsync: validatePathMutation, isPending: isValidating } =
    useMutation({
      mutationFn: validatePath,
    });

  const { data: installationStatus, isPending: isCheckingInstallation } =
    useQuery({
      queryKey: ["checkInstallationPath", validatedPath],
      queryFn: () => checkInstallationPath(validatedPath),
      enabled: validatedPath.trim() !== "",
      staleTime: 30000, // Cache for 30 seconds
    });

  useEffect(() => {
    let cancelled = false;

    const runValidation = async () => {
      const normalizedPath = debouncedPath.trim();
      if (!normalizedPath) {
        setValidation(undefined);
        setValidatedPath("");
        return;
      }

      setValidatedPath("");

      try {
        const result = await validatePathMutation({ path: normalizedPath });
        if (cancelled) {
          return;
        }
        setValidation(result);
        setValidatedPath(result.valid ? normalizedPath : "");
      } catch {
        if (cancelled) {
          return;
        }
        setValidation(undefined);
        setValidatedPath("");
      }
    };

    void runValidation();

    return () => {
      cancelled = true;
    };
  }, [debouncedPath, validatePathMutation]);

  const isLoading =
    isValidating || (isCheckingInstallation && validatedPath.trim() !== "");

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
