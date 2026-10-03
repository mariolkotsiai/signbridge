import { useNow } from '../hooks/useNow';
import { T } from '../strings';
import type { ResultMessage, SubtitleEntry } from '../types';

/** Πόση ώρα μένει στην οθόνη ο τελευταίος υπότιτλος πριν δώσει τη θέση του στο status hint. */
const SUBTITLE_TTL_MS = 4000;

interface Props {
  result: ResultMessage | null;
  subtitles: SubtitleEntry[];
  cameraOn: boolean;
}

/** Η «μπάρα υποτίτλων» πάνω στο βίντεο: τελευταίος υπότιτλος + ποσοστό εμπιστοσύνης. */
export function SubtitleBar({ result, subtitles, cameraOn }: Props) {
  const now = useNow(500);
  const latest = subtitles[0];
  const fresh = latest && now - latest.at < SUBTITLE_TTL_MS;

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
        hint = `${T.hintCollecting} ${result.buffer.filled}/${result.buffer.size}`;
        break;
      case 'uncertain':
        hint = T.hintUncertain;
        break;
      case 'sign':
        hint = result.stable_label ?? '';
        break;
      default:
        hint = T.hintIdle;
    }
  }
  return (
    <div className="subtitle-bar subtitle-hint" role="status" aria-live="polite">
      <span className="subtitle-text">{hint}</span>
    </div>
  );
}
