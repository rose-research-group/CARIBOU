/**
 * Pure helpers for review gates: the circle after a block that has a work
 * item, showing whether that attempt's work was approved. No Angular, no I/O.
 *
 * Reviews carry no attempt number, so a review belongs to the attempt whose
 * block was the latest one for the item at the review's turn: turn_start of
 * this block <= review.turn < turn_start of the next attempt's block.
 */
import { Block } from '../../../core/models/block.model';
import { WorkItemDetail, WorkItemReview } from '../../../core/models/session.model';
import { QcMode, canHumanReview } from './block-actions';

export type GateState = 'loading' | 'unavailable' | 'not_ready' | 'awaiting' | 'reviewing' | 'approved' | 'changes';

export const GATE_LABEL: Record<GateState, string> = {
  loading: 'loading',
  unavailable: 'work item unavailable',
  not_ready: 'not ready',
  awaiting: 'awaiting review',
  reviewing: 'in review',
  approved: 'approved',
  changes: 'changes requested',
};

/** Shape/glyph per state, so status never rests on colour alone. */
export const GATE_GLYPH: Record<GateState, string> = {
  loading: '…',
  unavailable: '!',
  not_ready: '',
  awaiting: '?',
  reviewing: '',
  approved: '✓',
  changes: '✗',
};

export interface Gate {
  state: GateState;
  /** The verdict-bearing reviews of this attempt, oldest first. */
  reviews: WorkItemReview[];
  /** The next attempt at the same work item, if one exists. */
  nextAttempt: Block | null;
  /**
   * Whether the connector after the gate is solid. Closed (false) only in
   * required-QC mode before approval; in optional mode the gate only informs.
   */
  open: boolean;
  /** Extra wording for the label (e.g. "superseded by attempt 2"). */
  note: string | null;
}

export interface GateInput {
  block: Block;
  /** All blocks of the session (to find the next attempt). */
  blocks: Block[];
  item: WorkItemDetail | null;
  itemError: string | null;
  qcMode: QcMode | null;
  /** An evaluator review of this item is running (started from this page). */
  evaluatorRunning: boolean;
}

/** The next attempt's block: same work item, attempt + 1. */
export function nextAttempt(block: Block, blocks: Block[]): Block | null {
  if (block.work_item_id === null) return null;
  return blocks.find(b => b.work_item_id === block.work_item_id && b.attempt === block.attempt + 1) ?? null;
}

/** The reviews recorded during this attempt that carry a verdict. */
export function attemptReviews(block: Block, next: Block | null, item: WorkItemDetail): WorkItemReview[] {
  return item.reviews.filter(r =>
    r.verdict !== null && r.error === null &&
    r.turn >= block.turn_start && (next === null || r.turn < next.turn_start));
}

/** The block's gate, or null for an implicit block (no work item, no gate). */
export function blockGate(input: GateInput): Gate | null {
  const { block, item, qcMode } = input;
  if (block.work_item_id === null) return null;
  const next = nextAttempt(block, input.blocks);
  const gate = (state: GateState, reviews: WorkItemReview[] = [], note: string | null = null): Gate => ({
    state, reviews, nextAttempt: next, note,
    open: qcMode !== 'required' || state === 'approved',
  });
  if (item === null) return gate(input.itemError !== null ? 'unavailable' : 'loading');
  const reviews = attemptReviews(block, next, item);
  const latest = reviews[reviews.length - 1];
  if (next === null && input.evaluatorRunning) return gate('reviewing', reviews);
  if (latest?.verdict === 'approve') return gate('approved', reviews);
  if (latest?.verdict === 'reject') return gate('changes', reviews);
  // The item's status is the latest attempt's; an earlier attempt with no
  // verdict was superseded without a review.
  if (next !== null) return gate('not_ready', reviews, `superseded by attempt ${next.attempt} (${next.block_id})`);
  if (canHumanReview(item.status, qcMode)) return gate('awaiting', reviews);
  return gate('not_ready', reviews, item.status);
}

/** "Review gate for blk-0003: awaiting review". */
export function gateAriaLabel(block: Block, gate: Gate): string {
  const note = gate.note ? ` (${gate.note})` : '';
  return `Review gate for ${block.block_id}: ${GATE_LABEL[gate.state]}${note}`;
}

/** `#gate:<block_id>` → the block id, or null for any other fragment. */
export function gateIdFromFragment(fragment: string | null): string | null {
  if (!fragment || !fragment.startsWith('gate:')) return null;
  const id = fragment.slice('gate:'.length);
  return id || null;
}
