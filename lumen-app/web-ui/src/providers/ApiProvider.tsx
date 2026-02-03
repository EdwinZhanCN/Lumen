import type { ReactNode } from 'react';

export function ApiProvider({ children }: { children: ReactNode }) {
  // Simple pass-through provider
  // Can be extended later for API context if needed
  return <>{children}</>;
}
