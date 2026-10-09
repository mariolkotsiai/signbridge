import { T } from '../strings';
import { S } from '../stringsSentence';
import type { PredictionPayload, ResultMessage, SubtitleEntry } from '../types';

interface Props {
  prediction: PredictionPayload | null;
  result: ResultMessage | null;
  roundTripMs: number;
  subtitles: SubtitleEntry[];
  threshold: number;
  sentenceMode?: boolean;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Ζωντανές πιθανότητες (top-k) με δείκτη ορίου, στατιστικά και ιστορικό υποτίτλων. */
export function PredictionPanel({ prediction, result, roundTripMs, subtitles, threshold, sentenceMode = false }: Props) {
  return (
    <aside className="panel">
      <section>
        <h2>{T.liveTop}</h2>
        {prediction ? (
          <ul className="bars">
            {prediction.top_k.map((k) => (
              <li key={k.label}>
                <div className="bar-head">
                  <span>{k.label}</span>
                  <span>{pct(k.confidence)}</span>
                </div>
                <div className="bar-track" aria-hidden="true">
                  <div
                    className={`bar-fill ${k.confidence >= threshold ? 'ok' : ''}`}
                    style={{ width: pct(k.confidence) }}
                  />
                  <div className="bar-threshold" style={{ left: pct(threshold) }} />
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">—</p>
        )}
      </section>

      <section className="stats">
        <div><span>{T.fps}</span><b>{result ? result.fps.toFixed(1) : '—'}</b></div>
        <div><span>{T.latency}</span><b>{result ? `${result.latency_ms.toFixed(0)} ms` : '—'}</b></div>
        <div><span>RTT</span><b>{roundTripMs ? `${roundTripMs.toFixed(0)} ms` : '—'}</b></div>
        <div>
          <span>{T.buffer}</span>
          <b>
            {result
              ? sentenceMode
                ? `${result.buffer.filled} ${S.frames}`
                : `${result.buffer.filled}/${result.buffer.size}`
              : '—'}
          </b>
        </div>
      </section>

      <section>
        <h2>{T.history}</h2>
        {subtitles.length === 0 ? (
          <p className="muted">{T.historyEmpty}</p>
        ) : (
          <ol className="history">
            {subtitles.map((s) => (
              <li key={s.id}>
                <span className="history-label">{s.label}</span>
                <span className="history-meta">
                  {pct(s.confidence)} · {new Date(s.at).toLocaleTimeString('el-GR')}
                </span>
              </li>
            ))}
          </ol>
        )}
      </section>
    </aside>
  );
}
