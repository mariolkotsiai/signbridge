import { useCallback, useEffect, useState } from 'react';
import { API_BASE } from '../config';
import type { HealthResponse } from '../types';

/** Διαβάζει το /api/health: ποια μοντέλα υπάρχουν, αν λείπει το hand_landmarker.task κ.λπ. */
export function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`, { cache: 'no-store' });
      if (!res.ok) throw new Error(String(res.status));
      setHealth((await res.json()) as HealthResponse);
      setUnreachable(false);
    } catch {
      setUnreachable(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 10_000);
    return () => window.clearInterval(id);
  }, [refresh]);

  return { health, unreachable, refresh };
}
