import { useEffect, useState, type RefObject } from 'react';
import { T } from '../strings';

export type CameraStatus = 'off' | 'starting' | 'on' | 'error';

function describeError(err: unknown): string {
  if (!navigator.mediaDevices?.getUserMedia) return T.cameraInsecure;
  const name = err instanceof DOMException ? err.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return T.cameraDenied;
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return T.cameraMissing;
  return `${T.cameraGeneric}: ${err instanceof Error ? err.message : String(err)}`;
}

/**
 * Ανοίγει την κάμερα όταν enabled=true και συνδέει το stream στο <video>.
 * Κλείνει πάντα τα tracks στο cleanup (σβήνει και η ένδειξη κάμερας του browser).
 */
export function useCamera(enabled: boolean, videoRef: RefObject<HTMLVideoElement | null>) {
  const [status, setStatus] = useState<CameraStatus>('off');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setStatus('off');
      return;
    }
    let cancelled = false;
    let stream: MediaStream | null = null;
    setStatus('starting');
    setError(null);

    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('mediaDevices unavailable');
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: 'user' },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play().catch(() => undefined); // το autoplay μπορεί να απορριφθεί· το muted+playsInline το καλύπτει
        }
        setStatus('on');
      } catch (err) {
        if (cancelled) return;
        setError(describeError(err));
        setStatus('error');
      }
    })();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      const video = videoRef.current;
      if (video) video.srcObject = null;
    };
  }, [enabled, videoRef]);

  return { status, error };
}
