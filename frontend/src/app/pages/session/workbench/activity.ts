/**
 * Pure helpers for the workbench's "agent is working" indicator: a mark
 * reduced from the latest events (what the agent is doing that the store's
 * other state doesn't say), and the one-line activity text. No Angular, no I/O.
 */
import { AgentEvent, AgentSwitchData, StatusChangeData, SystemMessageData } from '../../../core/models/events.model';

/** The latest notable event of the current turn. */
export type ActivityMark =
  | { kind: 'none' }
  | { kind: 'delegating'; to: string }
  | { kind: 'continuing'; step: number; total: number };

export const NO_ACTIVITY: ActivityMark = { kind: 'none' };

// streaming_runner.py: auto_continue_message(), category "Runner guidance".
const AUTO_CONTINUE = /^Continuing automatically\b.*\((\d+) of (\d+)\)\.?$/;

/**
 * The mark after one event. A delegation or an automatic continuation sets
 * it; the agent's next output (tokens, code, a message) or the end of the
 * turn clears it. Other events leave it as it is (same reference).
 */
export function reduceActivityMark(mark: ActivityMark, event: AgentEvent): ActivityMark {
  switch (event.type) {
    case 'agent_switch':
      return { kind: 'delegating', to: (event.data as AgentSwitchData).to_agent };
    case 'system_message': {
      const d = event.data as SystemMessageData;
      const m = d.category === 'Runner guidance' ? AUTO_CONTINUE.exec(d.content.trim()) : null;
      return m ? { kind: 'continuing', step: Number(m[1]), total: Number(m[2]) } : mark;
    }
    case 'token':
    case 'code_submitted':
    case 'message_complete':
      return mark.kind === 'none' ? mark : NO_ACTIVITY;
    case 'status_change': {
      const status = (event.data as StatusChangeData).status;
      return status === 'idle' || status === 'stopped' || status === 'error' ? NO_ACTIVITY : mark;
    }
    default:
      return mark;
  }
}

export interface ActivityInput {
  working: boolean;
  status: string;
  streaming: boolean;
  /** Code blocks submitted with no result yet. */
  pendingCode: number;
  mark: ActivityMark;
}

/** The activity line, e.g. "Running code (2)" or "Delegating to QC_metrics_agent". */
export function activityLine(a: ActivityInput): string {
  if (!a.working) return 'Waiting for your input';
  if (a.status === 'initializing') return 'Initializing…';
  if (a.status === 'recovering') return 'Recovering…';
  if (a.pendingCode > 0) return `Running code (${a.pendingCode})`;
  if (a.streaming) return 'Thinking…';
  if (a.mark.kind === 'delegating') return `Delegating to ${a.mark.to}`;
  if (a.mark.kind === 'continuing') return `Continuing automatically (${a.mark.step} of ${a.mark.total})`;
  return 'Thinking…';
}

/**
 * A server timestamp in ms. The server writes naive UTC ISO strings
 * (`datetime.utcnow().isoformat()`), so one with no offset is read as UTC.
 */
export function parseServerTime(timestamp: string): number {
  const iso = /(Z|[+-]\d\d:?\d\d)$/i.test(timestamp) ? timestamp : `${timestamp}Z`;
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) throw new Error(`Unparseable server timestamp: ${timestamp}`);
  return ms;
}

/** "42s", "3m 05s", "1h 02m". */
export function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}
