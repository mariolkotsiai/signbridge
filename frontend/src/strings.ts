// Όλα τα κείμενα του UI σε ένα σημείο (εύκολη μετάφραση / αλλαγή).
export const T = {
  title: 'SignBridge',
  subtitle: 'Αναγνώριση νοηματικής γλώσσας σε πραγματικό χρόνο',
  language: 'Γλώσσα',
  gsl: 'ΕΝΓ',
  asl: 'ASL',
  start: 'Έναρξη κάμερας',
  stop: 'Διακοπή',
  reset: 'Καθαρισμός',
  threshold: 'Όριο εμπιστοσύνης',
  noModel: 'χωρίς μοντέλο',

  statusOff: 'Εκτός λειτουργίας',
  statusConnecting: 'Σύνδεση…',
  statusStreaming: 'Ζωντανά',
  statusReconnecting: 'Επανασύνδεση…',
  statusError: 'Σφάλμα',

  hintIdle: 'Δείξε τα χέρια σου στην κάμερα',
  hintCollecting: 'Διαβάζω νόημα…',
  hintUncertain: 'Δεν είμαι σίγουρος',
  hintCameraOff: 'Πάτα «Έναρξη κάμερας» για να ξεκινήσεις',

  liveTop: 'Πιθανότερες αναγνωρίσεις',
  history: 'Ιστορικό υποτίτλων',
  historyEmpty: 'Οι υπότιτλοι θα εμφανιστούν εδώ.',
  confidence: 'Εμπιστοσύνη',
  latency: 'Καθυστέρηση',
  fps: 'FPS',
  buffer: 'Buffer',

  backendDown: 'Δεν βρέθηκε το backend. Τρέξε: uvicorn services.api.main:app',
  landmarkerMissing: 'Λείπει το hand_landmarker.task. Τρέξε: python scripts/download_models.py',
  cameraDenied: 'Δεν δόθηκε άδεια χρήσης κάμερας. Επίτρεψέ την από τον browser και δοκίμασε ξανά.',
  cameraMissing: 'Δεν βρέθηκε κάμερα.',
  cameraInsecure: 'Η κάμερα απαιτεί ασφαλές πλαίσιο: άνοιξε τη σελίδα από http://localhost ή https.',
  cameraGeneric: 'Αδυναμία εκκίνησης κάμερας',
} as const;
