/**
 * Ρυθμίσεις client. Από προεπιλογή χρησιμοποιούμε σχετικές διευθύνσεις, ώστε να δουλεύει
 * τόσο με τον Vite proxy (npm run dev) όσο και όταν το UI σερβίρεται από το ίδιο το FastAPI.
 */
export const STREAM = {
  /** Frames ανά δευτερόλεπτο που στέλνονται στο backend. 30 frames window ≈ 2 s στα 15 fps. */
  fps: 15,
  /** Ποιότητα JPEG (0..1). 0.7 ≈ 25-40 KB ανά frame στα 640x480. */
  jpegQuality: 0.7,
  /** Μέγιστο πλάτος frame που στέλνεται (το backend κάνει downscale ούτως ή άλλως). */
  maxWidth: 640,
  /** Αν δεν έρθει απάντηση σε τόσο χρόνο, θεωρούμε ότι χάθηκε και στέλνουμε το επόμενο frame. */
  responseTimeoutMs: 3000,
} as const;

export function wsUrl(language: string): string {
  const explicit = import.meta.env.VITE_WS_URL as string | undefined;
  const base =
    explicit && explicit.length > 0
      ? explicit
      : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/infer`;
  return `${base}?language=${encodeURIComponent(language)}`;
}

export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';
