/**
 * Pure helpers that join a block record to the rest of the session state:
 * its code actions (chat code items), its artifacts and its plots. No
 * Angular, no I/O. The block record is the source of truth: `action_ids` and
 * `artifact_paths` decide membership, the store's lists only supply detail.
 */
import { Artifact } from '../../../core/models/session.model';
import { Block, BlockStatus } from '../../../core/models/block.model';
import { ChatItem, CodeEvent } from '../../../core/state/session-state.model';

/** Status glyphs, the same as the CLI `/blocks` command. */
export const STATUS_GLYPH: Record<BlockStatus, string> = {
  running: '▶',
  ok: '✓',
  warn: '⚠',
  error: '✗',
};

export const STATUS_LABEL: Record<BlockStatus, string> = {
  running: 'Running',
  ok: 'OK',
  warn: 'Warning',
  error: 'Error',
};

/** One of a block's actions, with its code card if the chat has it. */
export interface BlockAction {
  actionId: string;
  /** 1-based position in the block's execution order. */
  position: number;
  failed: boolean;
  turn: number | null;
  codeEvent: CodeEvent | null;
}

/** One of a block's artifact paths, with its artifact if the list has it yet. */
export interface BlockArtifact {
  path: string;
  artifact: Artifact | null;
}

/**
 * The block's actions in execution order, each joined to the chat code item
 * whose `action_id` matches. An action with no code item (not yet received,
 * or recorded by a server that did not send action ids) has `codeEvent` null.
 */
export function blockActions(block: Block, chatItems: ChatItem[]): BlockAction[] {
  const byActionId = new Map<string, ChatItem>();
  for (const item of chatItems) {
    const actionId = item.kind === 'code' ? item.codeEvent?.submitted.action_id : undefined;
    if (actionId) byActionId.set(actionId, item);
  }
  const failed = new Set(block.failed_action_ids);
  return block.action_ids.map((actionId, i) => {
    const item = byActionId.get(actionId);
    return {
      actionId,
      position: i + 1,
      failed: failed.has(actionId),
      turn: item?.turn ?? null,
      codeEvent: item?.codeEvent ?? null,
    };
  });
}

/**
 * The block's artifact paths in first-seen order, each joined to the
 * artifact with that `path`. The artifact list is refetched after each
 * artifact event, so a path can briefly have no artifact (null).
 */
export function blockArtifacts(block: Block, artifacts: Artifact[]): BlockArtifact[] {
  const byPath = new Map(artifacts.map(artifact => [artifact.path, artifact]));
  return block.artifact_paths.map(path => ({ path, artifact: byPath.get(path) ?? null }));
}

/**
 * The block's plot artifacts in `artifact_paths` order. A path whose
 * artifact is not in the list yet is left out until it arrives.
 */
export function blockPlots(block: Block, artifacts: Artifact[]): Artifact[] {
  const plots: Artifact[] = [];
  for (const entry of blockArtifacts(block, artifacts)) {
    if (entry.artifact?.type === 'plot') plots.push(entry.artifact);
  }
  return plots;
}

/** `#block:<block_id>` → the block id, or null for any other fragment. */
export function blockIdFromFragment(fragment: string | null): string | null {
  if (!fragment || !fragment.startsWith('block:')) return null;
  const id = fragment.slice('block:'.length);
  return id || null;
}

/** "turn 3" or "turns 3–5". */
export function turnSpan(block: Block): string {
  return block.turn_start === block.turn_end
    ? `turn ${block.turn_start}`
    : `turns ${block.turn_start}–${block.turn_end}`;
}

/** "work item #2 · attempt 1", or "no work item" for implicit blocks. */
export function workItemLabel(block: Block): string {
  return block.work_item_id === null
    ? 'no work item'
    : `work item #${block.work_item_id} · attempt ${block.attempt}`;
}

export function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}
