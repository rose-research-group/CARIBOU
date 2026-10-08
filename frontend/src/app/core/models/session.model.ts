export type SessionStatus = 'initializing' | 'idle' | 'running' | 'stopped' | 'error' | 'recovering';
export type SessionMode = 'interactive' | 'auto';
export type RunMode = 'full_system' | 'single_agent' | 'one_shot';
export type SandboxType = 'singularity' | 'docker' | 'offline';
export type ArtifactType = 'plot' | 'data' | 'code' | 'report';

export type MemoryStrategy = 'full' | 'episodic' | 'agent_report' | 'none';
export type PythonEnvironmentKind = 'conda' | 'venv' | 'pyenv' | 'unknown';

export interface PythonEnvironmentCandidate {
  name: string;
  path: string;
  python_executable: string;
  kind: PythonEnvironmentKind;
  sources: string[];
}

export interface ResolvedPythonEnvironment {
  mode: 'bundled' | 'host';
  path: string | null;
  python_executable: string;
  kind: PythonEnvironmentKind | null;
  python_version: string | null;
  fingerprint: string | null;
}

export interface MemoryConfig {
  strategy: string;
  working_history_size: number | null;
  summarization_threshold: number | null;
  chunk_size: number | null;
}

export interface Session {
  id: string;
  name: string;
  status: SessionStatus;
  mode: SessionMode;
  run_mode: RunMode;
  agent_system: string;
  llm_backend: string;
  resolved_model: ResolvedModelInfo | null;
  evaluator_model: EvaluatorModelState;
  sandbox_type: SandboxType;
  python_environment: ResolvedPythonEnvironment;
  dataset_path: string;
  max_turns: number | null;
  current_turn: number;
  current_agent: string;
  created_at: string;
  updated_at: string;
  artifact_count: number;
  message_count: number;
  memory: MemoryConfig | null;
  can_evaluate: boolean;
  parent_session_id: string | null;
  forked_from_checkpoint_id: string | null;
  /** Set on a branch: the parent block it was branched from. */
  forked_from_block_id?: string | null;
  branch_restore_mode?: BranchRestoreMode | null;
  branch_instruction?: string | null;
  attempt_number: number;
  recovery_mode: RecoveryMode | null;
  recovery_status: RecoveryStatus;
  recovery_detail: string | null;
  recovery_phase: string | null;
  recovery_step: number;
  recovery_total_steps: number;
  recovery_substep: number | null;
  recovery_substep_total: number | null;
  checkpoint_turn: number | null;
  checkpoint_healthy: boolean;
  // The effective brief mode — session override if one was set, else the
  // blueprint's own default. Null until the agent system has loaded.
  brief_mode: 'off' | 'context' | 'seed_item' | null;
  phase: 'briefing' | 'execution';
  brief: SessionBriefFields | null;
  /** Messages queued while the agent was busy, in delivery order. */
  message_queue: QueuedMessage[];
}

/** A user message waiting on the server for the agent to become ready. */
export interface QueuedMessage {
  /** uuid4 hex. */
  id: string;
  content: string;
  /** The workbench block the message focuses, or null for a chat message. */
  block_id: string | null;
  /** ISO 8601. */
  created_at: string;
}

export interface SessionBriefFields {
  schema_version: string;
  deliverable: string;
  in_scope: string[];
  out_of_scope: string[];
  done_when: string[];
  precedent: string[];
  interface_delta: string;
  risks: string[];
  review_class: 'routine' | 'shared' | 'scientific' | 'docs';
  source: string | null;
  created_at: string;
  created_by: string;
}

export type RecoveryMode = 'smart' | 'literal_replay';

/** How a branch rebuilds the state at its block's entry checkpoint. */
export type BranchRestoreMode = 'replay' | 'checkpoint' | 'llm_regen';

/** `POST /api/sessions/{id}/blocks/{block_id}/branch` (extra fields are rejected). */
export interface BranchRequest {
  /** 1..4000 characters; the branch's first user turn. */
  instruction: string;
  restore_mode: BranchRestoreMode;
  name?: string;
  /** Required (true) for replay when the entry checkpoint has no fingerprint. */
  acknowledge_unverified?: boolean;
}

/** One direct child branch, from `GET /api/sessions/{id}/branches` (sorted by created_at). */
export interface BranchSummary {
  session_id: string;
  name: string;
  status: SessionStatus;
  recovery_status: RecoveryStatus;
  forked_from_block_id: string;
  branch_restore_mode: BranchRestoreMode;
  branch_instruction: string;
  created_at: string;
}
export type RecoveryStatus =
  | 'none'
  | 'awaiting_checkpoint'
  | 'recovering'
  | 'recovered'
  | 'partial'
  | 'failed'
  | 'accepted_partial';

export interface SessionResumeRequest {
  recovery_mode: RecoveryMode;
  target_mode?: SessionMode;
  additional_turns?: number;
  acknowledge_replay_risk?: boolean;
}

export interface SessionForkRequest extends SessionResumeRequest {
  name: string;
  llm_backend?: string;
  model_name?: string;
  ollama_model?: string;
  evaluator_model?: EvaluatorModelConfig;
  model_change_reason?: string;
  /** Omitted inherits the source; null explicitly selects the bundled environment. */
  python_environment_path?: string | null;
}

export interface ResolvedModelInfo {
  provider: string;
  model: string;
  parameters: Record<string, unknown>;
}

export interface EvaluatorModelConfig {
  mode: 'inherit_worker' | 'explicit';
  llm_backend?: string | null;
  model_name?: string | null;
  ollama_model?: string | null;
}

export interface EvaluatorModelState {
  selection: EvaluatorModelConfig;
  resolved_model: ResolvedModelInfo | null;
  revision: number;
}

export interface EvaluatorModelUpdateRequest {
  selection: EvaluatorModelConfig;
  expected_revision: number;
  reason?: string | null;
}

export interface SessionCreateRequest {
  name?: string;
  mode: SessionMode;
  run_mode: RunMode;
  agent_system: string;
  llm_backend: string;
  model_name?: string;
  ollama_model?: string;
  sandbox_type: SandboxType;
  python_environment_path?: string;
  dataset_path: string;
  reference_dataset_path?: string;
  max_turns?: number;
  initial_prompt?: string;
  memory_strategy?: MemoryStrategy;
  memory_working_history_size?: number;
  memory_summarization_threshold?: number;
  memory_chunk_size?: number;
  compress_memory?: boolean;
  agent_report_memory?: boolean;
  evaluator_model?: EvaluatorModelConfig;
  // undefined/omitted = use the blueprint's own brief_policy default.
  // An explicit value overrides it for this session only.
  brief_mode?: 'off' | 'context' | 'seed_item';
}

export interface OpenRouterModel {
  id: string;
  canonical_slug: string;
  name: string;
  context_length: number | null;
  pricing: Record<string, string>;
  supported_parameters: string[];
  description: string | null;
  expiration_date: string | null;
}

export interface OpenRouterCatalogue {
  models: OpenRouterModel[];
  fetched_at: number;
  stale: boolean;
  catalog_url: string;
}

export interface OpenRouterEndpoint {
  slug: string;
  name: string;
  context_length: number | null;
  pricing: Record<string, string>;
}

export interface OpenRouterEndpointsResponse {
  model_id: string;
  endpoints: OpenRouterEndpoint[];
  catalog_url: string;
}

export interface Message {
  id: string;
  session_id: string;
  turn: number;
  role: string;
  agent_name: string;
  content: string;
  timestamp: string;
  is_delegation: boolean;
}

export interface Artifact {
  id: string;
  session_id: string;
  turn: number;
  type: ArtifactType;
  filename: string;
  mime_type: string;
  size_bytes: number;
  created_at: string;
  local_path: string;
  download_url: string;
  /** Posix path relative to the session output_dir; the artifact's identity. */
  path: string;
  /** Code block that produced it (opaque), or null if produced outside one. */
  action_id: string | null;
  /** File mtime; changes when the same `path` is overwritten. */
  mtime_ns: number;
}

export interface CodeEvent {
  id: string;
  session_id: string;
  turn: number;
  agent_name: string;
  source: string;
  stdout: string;
  stderr: string;
  success: boolean;
  duration_ms: number;
}

export interface LLMBackend {
  id: string;
  provider: string;
  display_name: string;
  available: boolean;
  model_name?: string | null;
  thinking?: boolean | null;
  status?: string | null;
  message?: string | null;
  suggested_fix?: string | null;
}

export interface OllamaModelsResponse {
  host: string;
  running: boolean;
  models: string[];
  default_model: string;
  status: string;
  message: string;
  suggested_fix?: string | null;
}

export interface AgentBlueprint {
  name: string;
  description: string;
  agents: string[];
  has_rag: boolean;
  path: string;
  is_package_default: boolean;
}

export interface ServerStatus {
  version: string;
  sandbox_type: string;
  active_sessions: number;
}

export interface Dataset {
  filename: string;
  path: string;
  size_bytes: number;
  uploaded_at: string;
}

export interface ContextBreakdown {
  pinned_system: number;
  pivotal_code: number;
  summaries: number;
  working_user: number;
  working_assistant: number;
  working_system: number;
  total: number;
  total_full_history: number;
  global_messages?: number;
  agent_reports?: number;
  has_agent_prompt?: boolean;
  // Estimated token counts (see caribou/execution/token_utils.py) — approximate,
  // not an exact tokenizer count, but a much better proxy for context size
  // than raw message counts.
  pinned_system_tokens?: number;
  pivotal_code_tokens?: number;
  summaries_tokens?: number;
  working_user_tokens?: number;
  working_assistant_tokens?: number;
  working_system_tokens?: number;
  total_tokens?: number;
  total_full_history_tokens?: number;
  global_messages_tokens?: number;
  agent_reports_tokens?: number;
  agent_prompt_tokens?: number;
}

export interface MemoryState {
  strategy: string;
  config?: Record<string, number>;
  total_messages?: number;
  pinned_count?: number;
  summary_entries?: number;
  summarized_message_count?: number;
  unsummarized_count?: number;
  pivotal_code_count?: number;
  working_history_count?: number;
  context_estimate?: number;
  context_estimate_tokens?: number;
  total_full_history_tokens?: number;
  report_count?: number;
  global_message_count?: number;
  has_agent_prompt?: boolean;
  context_breakdown: ContextBreakdown;
}

export interface EvaluationResult {
  session_id: string;
  turn: number;
  evaluator_agent: string;
  evaluator_source: string;
  model: string;
  provider: string | null;
  evaluator_model: ResolvedModelInfo | null;
  evaluator_model_revision: number;
  provider_receipt: Record<string, unknown>;
  assessment: string;
  created_at: string;
}

export interface WorkItemSummary {
  id: number;
  title: string;
  status: 'Backlog' | 'Ready' | 'In progress' | 'In review' | 'Done';
  owner: string;
  created_turn: number;
  created_at: string;
  completed_turn: number | null;
  completed_at: string | null;
}

export interface WorkItemDetail extends WorkItemSummary {
  schema_version: string;
  session_id: string;
  origin_run_id: string;
  body: string;
  completion_summary: string | null;
  closed_turn: number | null;
  closed_at: string | null;
  transitions: WorkItemTransition[];
  reviews: WorkItemReview[];
  opening_commit: string | null;
  latest_commit: string | null;
  /** Where a human-opened ticket came from; null for agent-opened items (and v2 items). */
  anchor: WorkItemAnchor | null;
}

/** The workbench location a ticket was opened from. At least one field is non-null. */
export interface WorkItemAnchor {
  block_id: string | null;
  action_id: string | null;
  artifact_path: string | null;
}

/** `POST /api/sessions/{id}/work-items`: a ticket opened by the user. */
export interface WorkItemCreateRequest {
  title: string;
  body: string;
  /** An agent in the session's blueprint. */
  owner: string;
  anchor: WorkItemAnchor | null;
}

/** `POST /api/sessions/{id}/work-items/{n}/human-review`. */
export interface HumanReviewRequest {
  verdict: 'approve' | 'reject';
  /** Optional note for approve; required (non-empty) for reject. */
  assessment: string;
}

export interface WorkItemTransition {
  kind: string;
  actor: string;
  turn: number;
  timestamp: string;
  from_status: string | null;
  to_status: string;
  from_owner: string | null;
  to_owner: string;
}

export interface WorkItemReview {
  evaluator: string;
  turn: number;
  timestamp: string;
  verdict: 'approve' | 'reject' | null;
  assessment: string;
  provider_receipt: Record<string, unknown>;
  error: string | null;
}

export interface WorkItemReviewResult {
  item: WorkItemDetail;
  verdict: 'approve' | 'reject';
  assessment: string;
  provider_receipt: Record<string, unknown>;
}
