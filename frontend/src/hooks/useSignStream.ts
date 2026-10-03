import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import { STREAM, wsUrl } from '../config';
import type {
  ConnectionStatus,
  Language,
  PredictionPayload,
  ReadyMessage,
  ResultMessage,
  ServerMessage,
  SubtitleEntry,
} from '../types';

interface Options {
  videoRef: RefObject<HTMLVideoElement | null>;
  language: Language;
  /** true = ανοιχτή σύνδεση + αποστολή frames. */
  enabled: boolean;
}

const MAX_SUBTITLES = 50;
const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000];
// Κωδικοί κλεισίματος που ΔΕΝ έχει νόημα να επαναδοκιμαστούν αυτόματα (λάθος αίτημα, λείπει μοντέλο, κ.λπ.).
const FATAL_CODES = new Set([4400, 4404, 4500, 4501]);

/**
 * Συνδέει την κάμερα με το backend:
 *   <video> → canvas → JPEG → WebSocket (binary) → JSON αποτελέσματα.
 *
 * Έλεγχος ροής: στέλνουμε ΕΝΑ frame και περιμένουμε την απάντησή του πριν στείλουμε το επόμενο.
 * Έτσι, αν το backend είναι πιο αργό από τα 15 fps, δεν δημιουργείται ουρά και καθυστέρηση.
 */
export function useSignStream({ videoRef, language, enabled }: Options) {
  const [status, setStatus] = useState<ConnectionStatus>('off');
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState<ReadyMessage | null>(null);
  const [result, setResult] = useState<ResultMessage | null>(null);
  const [prediction, setPrediction] = useState<PredictionPayload | null>(null);
  const [subtitles, setSubtitles] = useState<SubtitleEntry[]>([]);
  const [roundTripMs, setRoundTripMs] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const subtitleId = useRef(0);

  useEffect(() => {
    if (!enabled) {
      setStatus('off');
      setReady(null);
      setResult(null);
      setPrediction(null);
      return;
    }

    let disposed = false;
    let attempt = 0;
    let timer: number | undefined; // setInterval της αποστολής frames
    let retryTimer: number | undefined;
    let inFlight = false;
    let sentAt = 0;
    const canvas = document.createElement('canvas');

    const stopSending = () => {
      if (timer !== undefined) window.clearInterval(timer);
      timer = undefined;
      inFlight = false;
    };

    const captureAndSend = (ws: WebSocket) => {
      const video = videoRef.current;
      if (!video || ws.readyState !== WebSocket.OPEN) return;
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) return;

      if (inFlight) {
        if (performance.now() - sentAt < STREAM.responseTimeoutMs) return;
        inFlight = false; // η απάντηση χάθηκε: συνεχίζουμε
      }

      const scale = Math.min(1, STREAM.maxWidth / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      // ΣΗΜΑΝΤΙΚΟ: στέλνουμε το frame ΜΗ καθρεπτισμένο (όπως τα βίντεο εκπαίδευσης).
      // Ο καθρέπτης στο UI είναι καθαρά οπτικός (CSS).
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      inFlight = true;
      sentAt = performance.now();
      canvas.toBlob(
        (blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
          else inFlight = false;
        },
        'image/jpeg',
        STREAM.jpegQuality,
      );
    };

    const connect = () => {
      if (disposed) return;
      setStatus(attempt === 0 ? 'connecting' : 'reconnecting');
      const ws = new WebSocket(wsUrl(language));
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;
      let fatal = false;

      ws.onmessage = (ev: MessageEvent) => {
        if (typeof ev.data !== 'string') return;
        let msg: ServerMessage;
        try {
          msg = JSON.parse(ev.data) as ServerMessage;
        } catch {
          return;
        }

        switch (msg.type) {
          case 'ready':
            attempt = 0;
            setError(null);
            setReady(msg);
            setStatus('streaming');
            stopSending();
            timer = window.setInterval(() => captureAndSend(ws), 1000 / STREAM.fps);
            break;
          case 'result':
            inFlight = false;
            setRoundTripMs(performance.now() - sentAt);
            setResult(msg);
            if (msg.prediction) {
              setPrediction(msg.prediction);
              const p = msg.prediction;
              if (p.committed && p.stable_label) {
                const entry: SubtitleEntry = {
                  id: ++subtitleId.current,
                  label: p.stable_label,
                  confidence: p.confidence,
                  at: Date.now(),
                };
                setSubtitles((prev) => [entry, ...prev].slice(0, MAX_SUBTITLES));
              }
            }
            break;
          case 'frame_error':
            inFlight = false; // μη μοιραίο: το επόμενο frame θα σταλεί κανονικά
            break;
          case 'error':
            setError(msg.message);
            break;
          default:
            break;
        }
      };

      ws.onclose = (ev: CloseEvent) => {
        stopSending();
        if (wsRef.current === ws) wsRef.current = null;
        if (disposed) return;
        fatal = FATAL_CODES.has(ev.code);
        if (fatal) {
          setStatus('error');
          return;
        }
        // Απρόσμενη διακοπή (π.χ. restart backend): επανασύνδεση με αυξανόμενη καθυστέρηση.
        const delay = RECONNECT_DELAYS_MS[Math.min(attempt, RECONNECT_DELAYS_MS.length - 1)];
        attempt += 1;
        setStatus('reconnecting');
        retryTimer = window.setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // Το onclose ακολουθεί πάντα· εκεί αποφασίζεται η επανασύνδεση.
        if (!fatal) setError((prev) => prev ?? 'Σφάλμα σύνδεσης με το backend');
      };
    };

    setError(null);
    setSubtitles([]);
    connect();

    return () => {
      disposed = true;
      stopSending();
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      const ws = wsRef.current;
      wsRef.current = null;
      ws?.close(1000);
    };
  }, [enabled, language, videoRef]);

  /** Αλλάζει threshold / min_votes στον server για αυτή τη σύνδεση. */
  const sendConfig = useCallback((cfg: { threshold?: number; min_votes?: number }) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'config', ...cfg }));
  }, []);

  /** Καθαρίζει buffer/ψηφοφορία στον server και το ιστορικό υποτίτλων τοπικά. */
  const reset = useCallback(() => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'reset' }));
    setSubtitles([]);
    setPrediction(null);
  }, []);

  return { status, error, ready, result, prediction, subtitles, roundTripMs, sendConfig, reset };
}
