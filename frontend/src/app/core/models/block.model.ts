/**
 * A block is the implementation that goes with a work item: one attempt at
 * it, or an implicit block for code run with no active work item. Mirrors the
 * backend's `caribou.block.v1` record (execution/blocks.py), which is the same
 * shape in blocks.json, `block_changed` events and the REST API.
 */
export type BlockStatus = 'running' | 'ok' | 'warn' | 'error';

export interface Block {
  schema_version: 'caribou.block.v1';
  /** Sequential per session, zero-padded (`blk-0001`), never reused. */
  block_id: string;
  session_id: string;
  /** 1-based creation order (matches the blk number). */
  index: number;
  work_item_id: number | null;
  /** >= 1; always 1 for implicit blocks. */
  attempt: number;
  implicit: boolean;
  title: string;
  /** Reserved for a later step; always null for now. */
  kind: null;
  /** Agent names in the order first seen in this block. */
  agents: string[];
  status: BlockStatus;
  turn_start: number;
  turn_end: number;
  /** make_action_id(...) values, in execution order. */
  action_ids: string[];
  /** Subset of action_ids whose code_result.success was false. */
  failed_action_ids: string[];
  /** Posix paths relative to output_dir (same as artifact `path`), first-seen order. */
  artifact_paths: string[];
  created_at: string;
  updated_at: string;
}

/** `GET /api/sessions/{id}/blocks`. `recorded` is false only for sessions with no blocks.json. */
export interface BlocksResponse {
  recorded: boolean;
  blocks: Block[];
}
