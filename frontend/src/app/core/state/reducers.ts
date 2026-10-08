/**
 * Pure reducers for session page state. Each takes the current value and an
 * event payload and returns the next value, returning the SAME reference when
 * nothing changes (so a signal update is a no-op). No Angular, no I/O: these
 * are the unit-testable core of SessionStore.
 */
import { Message, WorkItemSummary } from '../models/session.model';
import { Block } from '../models/block.model';
import {
  AgentEventEnvelope, AgentSwitchData, CodeSubmittedData, CodeResultData,
  ErrorData, RecoveryCompletedData, StatusChangeData, SystemMessageData,
} from '../models/events.model';
import { ChatItem, ErrorRecord, StatusEntry } from './session-state.model';

/** Turn used to order chat items (event turn, else the message's own). */
function itemTurn(item: ChatItem): number {
  return item.turn ?? item.message?.turn ?? 0;
}

/** Pending-code key: one code block of one turn. */
export function codeKey(turn: number, blockIndex: number): string {
  return `${turn}-${blockIndex}`;
}

export function withPendingCode(
  pending: Map<string, CodeSubmittedData>, turn: number, submitted: CodeSubmittedData,
): Map<string, CodeSubmittedData> {
  const next = new Map(pending);
  next.set(codeKey(turn, submitted.block_index), submitted);
  return next;
}

export function withoutPendingCode(
  pending: Map<string, CodeSubmittedData>, turn: number, blockIndex: number,
): Map<string, CodeSubmittedData> {
  const next = new Map(pending);
  next.delete(codeKey(turn, blockIndex));
  return next;
}

/**
 * Add a completed message. A message already present (by id) is left alone;
 * an optimistic `pending-*` message with the same role and content is
 * replaced in place; anything else is appended.
 */
export function upsertMessage(items: ChatItem[], message: Message): ChatItem[] {
  if (items.some(item => item.kind === 'message' && item.message?.id === message.id)) {
    return items;
  }
  const pendingIndex = items.findIndex(item =>
    item.kind === 'message' &&
    item.message?.id.startsWith('pending-') &&
    item.message.role === message.role &&
    item.message.content === message.content
  );
  if (pendingIndex >= 0) {
    return items.map((item, index) =>
      index === pendingIndex ? { kind: 'message', message, turn: message.turn } : item
    );
  }
  return [...items, { kind: 'message', message, turn: message.turn }];
}

/** The chat message a `system_message` event renders as. */
export function systemMessageFrom(ev: AgentEventEnvelope<SystemMessageData>): Message {
  const d = ev.data;
  return {
    id: d.id,
    session_id: ev.session_id,
    turn: ev.turn,
    role: 'system',
    agent_name: d.category || 'System',
    content: d.content,
    timestamp: ev.timestamp,
    is_delegation: false,
  };
}

/** The optimistic user message shown until the server's copy arrives. */
export function pendingUserMessage(sessionId: string, turn: number, content: string): ChatItem {
  return {
    kind: 'message',
    turn,
    message: {
      id: 'pending-' + Date.now(),
      session_id: sessionId,
      turn,
      role: 'user',
      agent_name: '',
      content,
      timestamp: new Date().toISOString(),
      is_delegation: false,
    },
  };
}

/**
 * Replace the chat's messages with the server's list, keeping event items
 * (code, delegations, errors, recoveries) and ordering everything by turn.
 * Returns null when the server has fewer messages than the chat (the cache is
 * newer), in which case the chat is kept as is.
 */
export function mergeServerMessages(items: ChatItem[], msgs: Message[]): ChatItem[] | null {
  if (msgs.length < items.filter(c => c.kind === 'message').length) return null;
  const messageItems: ChatItem[] = msgs.map(m => ({ kind: 'message' as const, message: m, turn: m.turn }));
  const eventItems = items.filter(item => item.kind !== 'message');
  return [...messageItems, ...eventItems].sort((a, b) => itemTurn(a) - itemTurn(b));
}

export function appendDelegation(items: ChatItem[], ev: AgentEventEnvelope<AgentSwitchData>): ChatItem[] {
  const d = ev.data;
  return [
    ...items,
    { kind: 'delegation', turn: ev.turn, delegation: { from: d.from_agent, to: d.to_agent, command: d.command } },
  ];
}

export function appendCodeSubmitted(items: ChatItem[], ev: AgentEventEnvelope<CodeSubmittedData>): ChatItem[] {
  return [...items, { kind: 'code', turn: ev.turn, codeEvent: { submitted: ev.data } }];
}

/**
 * Attach a code result to the newest card for the same turn and block index
 * that has no result yet. Matching the card itself (rather than requiring a
 * pending-code entry) covers a card restored from the cache mid-execution;
 * cards that already carry a result (replayed, cached) are left alone.
 */
export function attachCodeResult(
  items: ChatItem[], turn: number, result: CodeResultData,
): { items: ChatItem[]; attached: boolean } {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (item.kind === 'code' && item.codeEvent && !item.codeEvent.result &&
        item.codeEvent.submitted.block_index === result.block_index && item.turn === turn) {
      const updated = [...items];
      updated[i] = { ...item, codeEvent: { submitted: item.codeEvent.submitted, result } };
      return { items: updated, attached: true };
    }
  }
  return { items, attached: false };
}

export function errorRecordFrom(ev: AgentEventEnvelope<ErrorData>): ErrorRecord {
  const d = ev.data;
  return {
    code: d.code,
    message: d.message,
    fatal: d.fatal,
    timestamp: ev.timestamp,
    suggested_fix: d.suggested_fix ?? null,
  };
}

/** Recovery milestones are keyed by timestamp; a repeat is ignored. */
export function addRecovery(items: ChatItem[], ev: AgentEventEnvelope<RecoveryCompletedData>): ChatItem[] {
  const id = `recovery-${ev.timestamp.replace(/[^a-zA-Z0-9]/g, '-')}`;
  if (items.some(item => item.kind === 'recovery' && item.recovery?.id === id)) {
    return items;
  }
  return [
    ...items,
    { kind: 'recovery', turn: ev.turn, recovery: { ...ev.data, id, timestamp: ev.timestamp } },
  ];
}

/** Repeats of the latest status (same status and reason) bump its count. */
export function appendStatus(log: StatusEntry[], d: StatusChangeData, timestamp: string): StatusEntry[] {
  const last = log[log.length - 1];
  if (last && last.status === d.status && last.reason === d.reason) {
    return [...log.slice(0, -1), { ...last, count: last.count + 1 }];
  }
  return [...log, { status: d.status, reason: d.reason ?? null, timestamp, count: 1 }];
}

/** Replace a work item by id, keeping the list ordered by id. */
export function upsertWorkItem(items: WorkItemSummary[], item: WorkItemSummary): WorkItemSummary[] {
  return [...items.filter(existing => existing.id !== item.id), item].sort((a, b) => a.id - b.id);
}

/** Replace a block by block_id (the event is the newest state), ordered by index. */
export function upsertBlock(blocks: Block[], block: Block): Block[] {
  return [...blocks.filter(existing => existing.block_id !== block.block_id), block]
    .sort((a, b) => a.index - b.index);
}

/**
 * Merge a REST snapshot of blocks into the live list. A block already seen
 * through a `block_changed` event keeps the event's version unless the
 * snapshot is strictly newer (`updated_at`), since the snapshot may have been
 * read before that event.
 */
export function mergeBlockSnapshot(blocks: Block[], snapshot: Block[]): Block[] {
  const byId = new Map(blocks.map(block => [block.block_id, block]));
  for (const block of snapshot) {
    const live = byId.get(block.block_id);
    if (!live || live.updated_at < block.updated_at) byId.set(block.block_id, block);
  }
  return [...byId.values()].sort((a, b) => a.index - b.index);
}
