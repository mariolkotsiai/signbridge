import { useNow } from '../hooks/useNow';
import { T } from '../strings';
import { S } from '../stringsSentence';
import type { ResultMessage, SubtitleEntry } from '../types';

/** Πόση ώρα μένει στην οθόνη ο τελευταίος υπότιτλος πριν δώσει τη θέση του στο status hint. */
const SUBTITLE_TTL_MS = 4000;
/** Οι υπότιτλοι-προτάσεις είναι μακροί: ~100 ms ανά χαρακτήρα πάνω από ένα βασικό διάστημα. */
const ttlFor = (label: string) => Math.max(SUBTITLE_TTL_MS, 2000 + label.length * 100);

interface Props {
  result: ResultMessage | null;
  subtitles: SubtitleEntry[];
  cameraOn: boolean;
  /** Μοντέλο προτάσεων: ο υπότιτλος βγαίνει όταν κατεβαίνουν τα χέρια, όχι όταν γεμίζει ο buffer. */
  sentenceMode?: boolean;
}

/** Η «μπάρα υποτίτλων» πάνω στο βίντεο: τελευταίος υπότιτλος + ποσοστό εμπιστοσύνης. */
export function SubtitleBar({ result, subtitles, cameraOn, sentenceMode = false }: Props) {
  const now = useNow(500);
  const latest = subtitles[0];
  const fresh = latest && now - latest.at < ttlFor(latest.label);

  if (fresh) {
    return (
      <div className="subtitle-bar" role="status" aria-live="polite">
        <span className="subtitle-text">{latest.label}</span>
        <span className="subtitle-conf">{Math.round(latest.confidence * 100)}%</span>
      </div>
    );
  }

  let hint: string = T.hintCameraOff;
  if (cameraOn) {
    switch (result?.state) {
      case 'collecting':
        hint = sentenceMode
          ? `${S.hintCollecting} ${result.buffer.filled} ${S.frames}`
          : `${T.hintCollecting} ${result.buffer.filled}/${result.buffer.size}`;
        break;
      case 'uncertain':
        hint = T.hintUncertain;
        break;
      case 'sign':
        hint = result.stable_label ?? '';
        break;
      default:
        hint = sentenceMode ? S.hintIdle : T.hintIdle;
    }
  }
  return (
    <div className="subtitle-bar subtitle-hint" role="status" aria-live="polite">
      <span className="subtitle-text">{hint}</span>
    </div>
  );
}
