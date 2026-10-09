/**
 * Pure helpers for the Step continue mode's pause: after a successful
 * close_work_item the runner stops for user input with the guidance
 * "Stage closed: work item #N (block blk-000N). Review its gate, then send a
 * message or Continue." (streaming_runner.py, category "Runner guidance").
 * No Angular, no I/O.
 */
import { ChatItem } from '../../../core/state/session-state.model';

// execution/runner.py: stage_closed_message(); `(no code block)` when nothing ran under the item.
const STAGE_CLOSED = /^Stage closed: work item #(\d+) \((?:block ([^\s)]+)|no code block)\)/;

export interface StepPause {
  itemId: number;
  /** The closed item's latest block (its gate), or null when no code ran under it. */
  blockId: string | null;
}

/** The work item and block a "Stage closed:" guidance names, or null for any other text. */
export function parseStageClosed(content: string): StepPause | null {
  const m = STAGE_CLOSED.exec(content.trim());
  return m ? { itemId: Number(m[1]), blockId: m[2] ?? null } : null;
}

/**
 * The step pause the chat is in, or null: the latest Runner-guidance system
 * message is a "Stage closed:" one and nothing has moved on since (no user,
 * agent, code or delegation item after it). The caller adds the other half
 * of the condition, that the session is idle.
 */
export function stepPauseFrom(items: ChatItem[]): StepPause | null {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (item.kind === 'delegation' || item.kind === 'code') return null;
    const m = item.message;
    if (!m) continue;
    if (m.role === 'user' || m.role === 'assistant') return null;
    if (m.role === 'system' && m.agent_name === 'Runner guidance') return parseStageClosed(m.content);
  }
  return null;
}
