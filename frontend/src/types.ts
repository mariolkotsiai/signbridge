// Τύποι που αντικατοπτρίζουν ακριβώς τα JSON μηνύματα του FastAPI gateway (services/api/main.py).

export type Language = 'gsl' | 'asl' | 'gsl_sentences';

/** idle: δεν φαίνονται χέρια · collecting: γεμίζει ο buffer · sign: σταθερή αναγνώριση · uncertain: κάτω από το threshold */
export type SessionState = 'idle' | 'collecting' | 'sign' | 'uncertain';

/** Κανονικοποιημένο σημείο [x, y] στο διάστημα 0..1 (σε σχέση με το frame που στάλθηκε). */
export type Point = [number, number];
export type Hand = Point[]; // 21 σημεία

export interface TopK {
  label: string;
  confidence: number;
}

export interface PredictionPayload {
  label: string; // η πιο πιθανή κλάση (ανεξάρτητα από threshold)
  confidence: number;
  accepted: boolean; // confidence >= threshold
  top_k: TopK[];
  stable_label: string | null; // αποτέλεσμα ψηφοφορίας
  committed: boolean; // true ΜΙΑ φορά όταν εκδίδεται νέος υπότιτλος
}

export interface ReadyMessage {
  type: 'ready';
  language: Language;
  labels: string[];
  window_frames: number;
  min_frames: number;
  stride: number;
  threshold: number;
  min_votes: number;
  /** sign: λέξεις (κυλιόμενο παράθυρο) · sentence: πρόταση, απόφαση όταν κατεβαίνουν τα χέρια */
  mode?: 'sign' | 'sentence';
  seq_len?: number;
  sentence_end_frames?: number;
  sentence_max_frames?: number;
}

export interface ResultMessage {
  type: 'result';
  seq: number;
  language: Language;
  state: SessionState;
  hands: Hand[];
  buffer: { filled: number; size: number };
  stable_label: string | null;
  prediction: PredictionPayload | null; // null όταν σε αυτό το frame δεν έτρεξε inference
  latency_ms: number;
  fps: number;
}

export interface FrameErrorMessage {
  type: 'frame_error';
  code: string;
  message: string;
}

export interface ErrorMessage {
  type: 'error';
  code: string;
  message: string;
}

export interface ConfigAckMessage {
  type: 'config_ack';
  threshold: number;
  min_votes: number;
}

export type ServerMessage =
  | ReadyMessage
  | ResultMessage
  | FrameErrorMessage
  | ErrorMessage
  | ConfigAckMessage
  | { type: 'pong' };

export interface SubtitleEntry {
  id: number;
  label: string;
  confidence: number;
  at: number; // Date.now()
}

export type ConnectionStatus = 'off' | 'connecting' | 'streaming' | 'reconnecting' | 'error';

export interface ModelInfo {
  available: boolean;
  num_classes?: number;
  labels?: string[];
  mode?: 'sign' | 'sentence';
  seq_len?: number;
  error?: string;
}

export interface HealthResponse {
  status: string;
  active_sessions: number;
  max_sessions: number;
  landmarker_model: boolean;
  defaults: { threshold: number; window_frames: number };
  models: Record<Language, ModelInfo>;
}
