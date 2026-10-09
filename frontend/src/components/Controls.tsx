import { T } from '../strings';
import { S } from '../stringsSentence';
import type { HealthResponse, Language } from '../types';

interface Props {
  language: Language;
  onLanguage: (l: Language) => void;
  running: boolean;
  onToggle: () => void;
  onReset: () => void;
  threshold: number;
  onThreshold: (v: number) => void;
  health: HealthResponse | null;
}

const LANGS: Language[] = ['gsl', 'asl', 'gsl_sentences'];

// Ετικέτες που δεν υπάρχουν στο strings.ts (το μοντέλο προτάσεων).
const LABEL_OVERRIDE: Partial<Record<Language, string>> = { gsl_sentences: S.langSentences };

export function Controls({ language, onLanguage, running, onToggle, onReset, threshold, onThreshold, health }: Props) {
  return (
    <div className="controls">
      <div className="control-group" role="radiogroup" aria-label={T.language}>
        <span className="control-label">{T.language}</span>
        {LANGS.map((l) => {
          const available = health ? health.models[l]?.available : true;
          return (
            <button
              key={l}
              type="button"
              role="radio"
              aria-checked={language === l}
              className={`chip ${language === l ? 'active' : ''}`}
              onClick={() => onLanguage(l)}
              title={available ? undefined : health?.models[l]?.error}
            >
              {LABEL_OVERRIDE[l] ?? T[l as 'gsl' | 'asl']}
              {!available && <small> · {T.noModel}</small>}
            </button>
          );
        })}
      </div>

      <label className="control-group slider">
        <span className="control-label">
          {T.threshold}: <b>{Math.round(threshold * 100)}%</b>
        </span>
        <input
          type="range"
          min={30}
          max={99}
          step={1}
          value={Math.round(threshold * 100)}
          onChange={(e) => onThreshold(Number(e.target.value) / 100)}
        />
      </label>

      <div className="control-group">
        <button type="button" className={`btn ${running ? 'danger' : 'primary'}`} onClick={onToggle}>
          {running ? T.stop : T.start}
        </button>
        <button type="button" className="btn" onClick={onReset} disabled={!running}>
          {T.reset}
        </button>
      </div>
    </div>
  );
}
