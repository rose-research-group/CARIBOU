/**
 * Pure helpers for branching from a block: which restore modes a block's
 * entry checkpoint supports (contract section 4 / A3), the request body, the
 * alignment of a branch's blocks under the parent's columns (the Paths row)
 * and the wording of a branch's restore progress. No Angular, no I/O.
 */
import { Block } from '../../../core/models/block.model';
import {
  BranchRequest, BranchRestoreMode, RecoveryStatus, Session, SessionStatus,
} from '../../../core/models/session.model';
import { RerunLinks, Stage, stages } from './stages';

export const BRANCH_INSTRUCTION_MAX = 4000;

/** Source statuses the server accepts a branch from. */
const BRANCHABLE_STATUSES: ReadonlySet<SessionStatus> = new Set<SessionStatus>(['idle', 'stopped', 'error']);

export const RESTORE_MODE_LABEL: Record<BranchRestoreMode, string> = {
  replay: 'Replay',
  checkpoint: 'Checkpoint',
  llm_regen: 'LLM rebuild',
};

const RESTORE_MODE_EXPLANATION: Record<BranchRestoreMode, string> = {
  replay: 'Load the original dataset and re-run every recorded action before this block, then check the data matches.',
  checkpoint: 'Load the dataset saved when this block started. Only adata is restored; other Python variables are not.',
  llm_regen: 'Load the dataset saved when this block started, then have the model rebuild the other Python state.',
};

const MODES: BranchRestoreMode[] = ['replay', 'checkpoint', 'llm_regen'];

/** One restore-mode radio: enabled, or disabled with the reason. */
export interface RestoreModeOption {
  mode: BranchRestoreMode;
  label: string;
  explanation: string;
  enabled: boolean;
  /** Why the option is disabled; null when enabled. */
  reason: string | null;
  /** Replay with no fingerprint: the user must accept an unverified replay. */
  needsAcknowledgement: boolean;
}

/** The contract's wording for a block with no entry checkpoint. */
export function noEntryReason(block: Block): string {
  return `Block ${block.block_id} was recorded before branching support; it has no entry checkpoint.`;
}

/**
 * Why no branch can be made from this block in this session right now, or
 * null. Inherited blocks are immutable in a branch; the source must not be
 * running.
 */
export function branchUnavailableReason(block: Block, sourceStatus: SessionStatus | null): string | null {
  if (block.inherited_from != null) {
    return `Block ${block.block_id} is inherited; branch from it in session ${block.inherited_from.session_id}.`;
  }
  if (sourceStatus === null) return 'The session is not loaded.';
  if (!BRANCHABLE_STATUSES.has(sourceStatus)) {
    return `The session is ${sourceStatus}. Branching opens once it is idle, stopped or in error.`;
  }
  return null;
}

/**
 * The three restore modes for a block, following its `entry`:
 * - no entry or no checkpoint id: all disabled (pre-branching block);
 * - replay: needs the checkpoint; an absent fingerprint needs acknowledgement;
 * - checkpoint and llm_regen: need a complete checkpoint (its dataset saved).
 */
export function restoreModeOptions(block: Block): RestoreModeOption[] {
  const entry = block.entry;
  const hasCheckpoint = entry != null && entry.checkpoint_id != null;
  return MODES.map(mode => {
    let reason: string | null = null;
    if (!hasCheckpoint) {
      reason = noEntryReason(block);
    } else if (mode !== 'replay' && entry.checkpoint_complete !== true) {
      reason = 'The entry checkpoint is incomplete: its dataset was not saved, so there is nothing to load.';
    }
    return {
      mode,
      label: RESTORE_MODE_LABEL[mode],
      explanation: RESTORE_MODE_EXPLANATION[mode],
      enabled: reason === null,
      reason,
      needsAcknowledgement: mode === 'replay' && hasCheckpoint && entry.fingerprint == null,
    };
  });
}

/** The first enabled mode (the form's default), or null when none is. */
export function defaultRestoreMode(options: RestoreModeOption[]): BranchRestoreMode | null {
  return options.find(o => o.enabled)?.mode ?? null;
}

export interface BranchFormValue {
  instruction: string;
  mode: BranchRestoreMode | null;
  name: string;
  acknowledgeUnverified: boolean;
}

/** Why the form cannot be submitted, or null. */
export function branchFormError(value: BranchFormValue, options: RestoreModeOption[]): string | null {
  const instruction = value.instruction.trim();
  if (!instruction) return 'Write the instruction the branch starts with.';
  if (instruction.length > BRANCH_INSTRUCTION_MAX) {
    return `The instruction is ${instruction.length} characters; the limit is ${BRANCH_INSTRUCTION_MAX}.`;
  }
  if (value.mode === null) return 'Choose a restore mode.';
  const option = options.find(o => o.mode === value.mode);
  if (!option) throw new Error(`Unknown restore mode ${value.mode}.`);
  if (!option.enabled) return option.reason;
  if (option.needsAcknowledgement && !value.acknowledgeUnverified) {
    return 'Accept the unverified replay, or choose another mode.';
  }
  return null;
}

/** The request body. A blank name is omitted (the server names the branch). */
export function buildBranchRequest(value: BranchFormValue, options: RestoreModeOption[]): BranchRequest {
  const error = branchFormError(value, options);
  if (error !== null) throw new Error(error);
  const request: BranchRequest = {
    instruction: value.instruction.trim(),
    restore_mode: value.mode!,
  };
  const name = value.name.trim();
  if (name) request.name = name;
  const option = options.find(o => o.mode === value.mode)!;
  if (option.needsAcknowledgement) request.acknowledge_unverified = true;
  return request;
}

// ── Paths row: lane alignment ──

/** One of a branch's stages placed under the parent's step columns (1-based). */
export interface LaneCell extends Stage {
  /** The stage's latest block. */
  block: Block;
  column: number;
  inherited: boolean;
}

export type LaneLayout =
  | { ok: true; cells: LaneCell[]; lastColumn: number }
  | { ok: false; error: string };

/**
 * Place a branch's blocks (its own GET /blocks) under the current session's
 * step columns (`currentColumns`: block id → column, every version of every
 * stage). Inherited copies sit at the column of the block they were copied
 * from (copies of one column make one cell); the branch's own blocks are
 * grouped into stages by the branch's rerun links and start at the column
 * of the block it branched from, in first-block order (they may run past
 * the parent's last column). A copy or branch point the current blocks
 * don't contain is a data inconsistency and is reported, not guessed.
 */
export function laneLayout(
  currentColumns: ReadonlyMap<string, number>,
  laneBlocks: Block[],
  laneLinks: RerunLinks,
  currentSessionId: string,
  forkedFromBlockId: string,
): LaneLayout {
  const start = currentColumns.get(forkedFromBlockId);
  if (start === undefined) {
    return { ok: false, error: `Branch point ${forkedFromBlockId} is not among this session's blocks.` };
  }
  const sorted = [...laneBlocks].sort((a, b) => a.index - b.index);
  const inheritedByColumn = new Map<number, LaneCell>();
  const own: Block[] = [];
  for (const block of sorted) {
    const origin = block.inherited_from;
    if (origin == null) {
      own.push(block);
      continue;
    }
    const column = origin.session_id === currentSessionId ? currentColumns.get(origin.block_id) : undefined;
    if (column === undefined) {
      return {
        ok: false,
        error: `Inherited block ${block.block_id} came from ${origin.session_id}/${origin.block_id}, which is not in this session.`,
      };
    }
    const cell = inheritedByColumn.get(column);
    if (cell) {
      cell.versions.push(block);
      cell.latest = block;
      cell.block = block;
    } else {
      inheritedByColumn.set(column, {
        key: `inherited:${column}`, rootItemId: null, versions: [block], latest: block, block, column, inherited: true,
      });
    }
  }
  const cells: LaneCell[] = [...inheritedByColumn.values()];
  stages(own, laneLinks).forEach((stage, i) => {
    cells.push({ ...stage, block: stage.latest, column: start + i, inherited: false });
  });
  const lastColumn = cells.reduce((max, c) => Math.max(max, c.column), 0);
  return { ok: true, cells, lastColumn };
}

// ── Branch banner ──

export const RECOVERY_STATUS_LABEL: Record<RecoveryStatus, string> = {
  none: 'not restored',
  awaiting_checkpoint: 'waiting for checkpoint',
  recovering: 'restoring',
  recovered: 'restored',
  partial: 'partially restored',
  failed: 'restore failed',
  accepted_partial: 'partial restore accepted',
};

/** Whether the session is a branch (a fork made from a block). */
export function isBranch(session: Session): boolean {
  return session.forked_from_block_id != null;
}

/** "Step 3 of 8 · detail", for a branch whose restore is under way. */
export function restoreProgressText(session: Session): string {
  const total = session.recovery_total_steps;
  const step = total > 0 ? `Step ${session.recovery_step} of ${total}` : 'Starting';
  const sub = session.recovery_substep != null && session.recovery_substep_total != null
    ? ` (${session.recovery_substep} / ${session.recovery_substep_total})`
    : '';
  const detail = session.recovery_detail ? ` · ${session.recovery_detail}` : '';
  return `${step}${sub}${detail}`;
}

/** Restore progress as a 0–100 percentage; 0 when the total is unknown. */
export function restorePercent(session: Session): number {
  const total = session.recovery_total_steps;
  if (!total) return 0;
  return Math.max(0, Math.min(100, (session.recovery_step / total) * 100));
}

/** A short session id for labels. */
export function shortId(id: string): string {
  return id.slice(0, 8);
}
