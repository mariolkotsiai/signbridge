import { useEffect, useRef, useState } from 'react';
import { Controls } from './components/Controls';
import { PredictionPanel } from './components/PredictionPanel';
import { SubtitleBar } from './components/SubtitleBar';
import { VideoStage } from './components/VideoStage';
import { useCamera } from './hooks/useCamera';
import { useHealth } from './hooks/useHealth';
import { useSignStream } from './hooks/useSignStream';
import { T } from './strings';
import type { ConnectionStatus, Language } from './types';

const STATUS_TEXT: Record<ConnectionStatus, string> = {
  off: T.statusOff,
  connecting: T.statusConnecting,
  streaming: T.statusStreaming,
  reconnecting: T.statusReconnecting,
  error: T.statusError,
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [language, setLanguage] = useState<Language>('gsl');
  const [running, setRunning] = useState(false);
  const [threshold, setThreshold] = useState(0.7);

  const { health, unreachable, refresh } = useHealth();
  const camera = useCamera(running, videoRef);
  const stream = useSignStream({ videoRef, language, enabled: running });

  // Αρχική τιμή του ορίου από τον server (SIGNBRIDGE_THRESHOLD), μία φορά.
  const thresholdInit = useRef(false);
  useEffect(() => {
    if (health && !thresholdInit.current) {
      thresholdInit.current = true;
      setThreshold(health.defaults.threshold);
    }
  }, [health]);

  // Κάθε νέα σύνδεση ξεκινά με το προεπιλεγμένο όριο του server: ξαναστέλνουμε αυτό που διάλεξε ο χρήστης.
  const { sendConfig } = stream;
  useEffect(() => {
    if (stream.status !== 'streaming') return;
    const id = window.setTimeout(() => sendConfig({ threshold }), 150); // debounce slider
    return () => window.clearTimeout(id);
  }, [threshold, stream.status, stream.ready, sendConfig]);

  // Μετά από σφάλμα (π.χ. έλειπε μοντέλο) ανανεώνουμε το health ώστε να φανεί η τρέχουσα κατάσταση.
  useEffect(() => {
    if (stream.status === 'error') void refresh();
  }, [stream.status, refresh]);

  const errorMessage = camera.error ?? stream.error ?? (unreachable ? T.backendDown : null) ??
    (health && !health.landmarker_model ? T.landmarkerMissing : null);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>{T.title}</h1>
          <p className="muted">{T.subtitle}</p>
        </div>
        <span className={`status status-${stream.status}`}>{STATUS_TEXT[stream.status]}</span>
      </header>

      <Controls
        language={language}
        onLanguage={setLanguage}
        running={running}
        onToggle={() => setRunning((r) => !r)}
        onReset={stream.reset}
        threshold={threshold}
        onThreshold={setThreshold}
        health={health}
      />

      {errorMessage && <div className="alert" role="alert">{errorMessage}</div>}

      <main className="layout">
        <VideoStage videoRef={videoRef} hands={stream.result?.hands ?? []}>
          <SubtitleBar result={stream.result} subtitles={stream.subtitles} cameraOn={running} />
        </VideoStage>
        <PredictionPanel
          prediction={stream.prediction}
          result={stream.result}
          roundTripMs={stream.roundTripMs}
          subtitles={stream.subtitles}
          threshold={threshold}
        />
      </main>
    </div>
  );
}
