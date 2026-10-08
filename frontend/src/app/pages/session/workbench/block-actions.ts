/**
 * Pure helpers for the workbench's steer-and-annotate actions (Tier 1): the
 * block-scoped message format the agent reads as plain text, which work-item
 * states accept a human review, anchor labels and HTTP error text. No
 * Angular, no I/O.
 */
import { HttpErrorResponse } from '@angular/common/http';
import { Block } from '../../../core/models/block.model';
import { WorkItemAnchor, WorkItemSummary } from '../../../core/models/session.model';
import { WorkItemPolicyConfig } from '../../../core/models/blueprint.model';

export type QcMode = WorkItemPolicyConfig['qc_mode'];

/** Something in the block the user points the agent at. */
export type BlockReference =
  | { kind: 'code'; actionId: string }
  | { kind: 'artifact'; path: string };

/** Stable key for a reference (selection sets, `track`). */
export function referenceKey(ref: BlockReference): string {
  return ref.kind === 'code' ? `code:${ref.actionId}` : `artifact:${ref.path}`;
}

/**
 * The header line of a block-scoped message:
 * `[Workbench · block blk-0003 "Quality control per batch" · work item #2, attempt 1]`,
 * or `... · no work item]` for an implicit block.
 */
export function blockMessageHeader(block: Block): string {
  const item = block.work_item_id === null
    ? 'no work item'
    : `work item #${block.work_item_id}, attempt ${block.attempt}`;
  return `[Workbench · block ${block.block_id} "${block.title}" · ${item}]`;
}

/** `Referenced: code <action_id>; artifact <path>`, in the given order. */
export function referencedLine(refs: BlockReference[]): string {
  const parts = refs.map(r => r.kind === 'code' ? `code ${r.actionId}` : `artifact ${r.path}`);
  return `Referenced: ${parts.join('; ')}`;
}

/**
 * References in a stable order: code in the block's execution order, then
 * artifacts in its first-seen order. A reference not in the block raises.
 */
export function orderedReferences(block: Block, refs: BlockReference[]): BlockReference[] {
  const rank = (r: BlockReference): number => {
    const i = r.kind === 'code' ? block.action_ids.indexOf(r.actionId) : block.artifact_paths.indexOf(r.path);
    if (i < 0) throw new Error(`Reference ${referenceKey(r)} is not part of block ${block.block_id}.`);
    return r.kind === 'code' ? i : block.action_ids.length + i;
  };
  return [...refs].sort((a, b) => rank(a) - rank(b));
}

/**
 * The full block-scoped message (contract section E): the header, a
 * `Referenced:` line only when something is selected, then the user's text.
 */
export function formatBlockMessage(block: Block, refs: BlockReference[], text: string): string {
  const body = text.trim();
  if (!body) throw new Error('A block message needs text.');
  const lines = [blockMessageHeader(block)];
  if (refs.length > 0) lines.push(referencedLine(orderedReferences(block, refs)));
  lines.push(body);
  return lines.join('\n');
}

/** The text of the message sent after a successful "Request changes". */
export function sentBackText(itemId: number, assessment: string): string {
  return `Work item #${itemId} was sent back for changes: ${assessment.trim()}`;
}

/**
 * Whether the backend accepts a human review of an item in this status:
 * Done in optional-QC mode, In review in required mode. With the mode
 * unknown (null), both statuses are offered and a 409 explains a mismatch.
 */
export function canHumanReview(status: WorkItemSummary['status'], qcMode: QcMode | null): boolean {
  if (qcMode === 'optional') return status === 'Done';
  if (qcMode === 'required') return status === 'In review';
  return status === 'Done' || status === 'In review';
}

/** "From block blk-0003 · code <action_id> · artifact <path>". */
export function anchorLabel(anchor: WorkItemAnchor): string {
  const parts: string[] = [];
  if (anchor.block_id) parts.push(`block ${anchor.block_id}`);
  if (anchor.action_id) parts.push(`code ${anchor.action_id}`);
  if (anchor.artifact_path) parts.push(`artifact ${anchor.artifact_path}`);
  return `From ${parts.join(' · ')}`;
}

/**
 * A readable message for a failed request: FastAPI's `detail` (a string for
 * HTTPException, a list of field errors for a 422), else the HTTP message.
 */
export function httpErrorMessage(err: unknown): string {
  const detail = httpErrorDetail(err);
  return err instanceof HttpErrorResponse && detail.fromServer ? `${err.status}: ${detail.text}` : detail.text;
}

/**
 * The server's `detail` exactly as sent (a 422's field errors joined), with
 * no status prefix; `fromServer` is false when there was no detail and the
 * text is the HTTP or JS error message instead.
 */
export function httpErrorDetail(err: unknown): { text: string; fromServer: boolean } {
  if (err instanceof HttpErrorResponse) {
    const detail = (err.error as { detail?: unknown } | null)?.detail;
    if (typeof detail === 'string' && detail) return { text: detail, fromServer: true };
    if (Array.isArray(detail) && detail.length > 0) {
      const msgs = detail.map(d => {
        const e = d as { loc?: unknown[]; msg?: string };
        const field = Array.isArray(e.loc) ? e.loc.filter(l => l !== 'body').join('.') : '';
        return field ? `${field}: ${e.msg}` : String(e.msg ?? JSON.stringify(d));
      });
      return { text: msgs.join('; '), fromServer: true };
    }
    return { text: err.message, fromServer: false };
  }
  if (err instanceof Error) return { text: err.message, fromServer: false };
  return { text: String(err), fromServer: false };
}
