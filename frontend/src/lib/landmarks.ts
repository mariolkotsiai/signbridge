import type { Hand } from '../types';

// Τοπολογία των 21 σημείων χεριού του MediaPipe (ίδια με το HAND_CONNECTIONS του backend).
export const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/**
 * Σχεδιάζει τα χέρια πάνω σε canvas. Οι συντεταγμένες είναι κανονικοποιημένες (0..1),
 * άρα ταιριάζουν σε οποιοδήποτε μέγεθος εμφάνισης του βίντεο.
 */
export function drawHands(canvas: HTMLCanvasElement, hands: Hand[]): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);

  ctx.lineWidth = Math.max(2, w / 300);
  ctx.lineCap = 'round';
  hands.forEach((hand, i) => {
    const color = i === 0 ? '#34d399' : '#60a5fa';
    ctx.strokeStyle = color;
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = hand[a];
      const pb = hand[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa[0] * w, pa[1] * h);
      ctx.lineTo(pb[0] * w, pb[1] * h);
      ctx.stroke();
    }
    ctx.fillStyle = '#f87171';
    for (const p of hand) {
      ctx.beginPath();
      ctx.arc(p[0] * w, p[1] * h, Math.max(3, w / 190), 0, Math.PI * 2);
      ctx.fill();
    }
  });
}
