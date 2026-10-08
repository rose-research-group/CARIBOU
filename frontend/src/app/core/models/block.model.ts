/**
 * A block is the implementation that goes with a work item: one attempt at
 * it, or an implicit block for code run with no active work item. Mirrors the
 * backend's `caribou.block.v2` record (execution/blocks.py), which is the same
 * shape in blocks.json, `block_changed` events and the REST API. v1 records
 * (written before branching) have no `entry` / `inherited_from`; the server
 * reports them as null, a cached copy may lack the keys entirely.
 */
export type BlockStatus = 'running' | 'ok' | 'warn' | 'error';

/** The sandbox's `adata` shape when a checkpoint was captured. */
export interface DatasetFingerprint {
  n_obs: number;
  n_vars: number;
  obs_keys: string[];
  var_keys: string[];
  obsm_keys: string[];
  layers_keys: string[];
}

/** The checkpoint captured when the block was created, before its first action. */
export interface BlockEntry {
  turn: number;
  checkpoint_id: string | null;
  /** Whether the checkpoint's dataset was saved (checkpoint / llm_regen need it). */
  checkpoint_complete: boolean;
  /** Null when the fingerprint could not be computed (replay is then unverified). */
  fingerprint: DatasetFingerprint | null;
  work_items_commit: string | null;
}

/** Where an inherited block was copied from (a branch's prefix). */
export interface BlockOrigin {
  session_id: string;
  block_id: string;
}

export interface Block {
  schema_version: 'caribou.block.v1' | 'caribou.block.v2';
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
  /** v2: the block-entry checkpoint; null for v1 and CLI blocks. */
  entry?: BlockEntry | null;
  /** v2: set on a branch's copy of a parent block; such blocks are immutable. */
  inherited_from?: BlockOrigin | null;
}

/** `GET /api/sessions/{id}/blocks`. `recorded` is false only for sessions with no blocks.json. */
export interface BlocksResponse {
  recorded: boolean;
  blocks: Block[];
}
