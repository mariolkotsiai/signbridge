import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import { drawHands } from '../lib/landmarks';
import type { Hand } from '../types';

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  hands: Hand[];
  children?: ReactNode; // overlays (υπότιτλοι, badges) που ΔΕΝ πρέπει να καθρεπτίζονται
}

/**
 * Βίντεο κάμερας + canvas με τα landmarks. Ο καθρέπτης (selfie view) εφαρμόζεται με CSS και στα
 * δύο μαζί, ώστε να μένουν ευθυγραμμισμένα, ενώ το κείμενο των υποτίτλων (children) μένει ανέγγιχτο.
 */
export function VideoStage({ videoRef, hands, children }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [aspect, setAspect] = useState('4 / 3');

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas) return;
    const w = video?.videoWidth || 640;
    const h = video?.videoHeight || 480;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    drawHands(canvas, hands);
  }, [hands, videoRef]);

  return (
    <div className="stage" style={{ aspectRatio: aspect }}>
      <div className="mirror">
        <video
          ref={videoRef}
          muted
          playsInline
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            if (v.videoWidth && v.videoHeight) setAspect(`${v.videoWidth} / ${v.videoHeight}`);
          }}
        />
        <canvas ref={canvasRef} />
      </div>
      {children}
    </div>
  );
}
