import { useEffect, useState } from 'react';

/** Επιστρέφει Date.now() που ανανεώνεται κάθε `intervalMs` (για υπότιτλους που «σβήνουν» μετά από λίγο). */
export function useNow(intervalMs = 500): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}
