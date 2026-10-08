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
  parent_session_id: string | null;
  forked_from_checkpoint_id: string | null;
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
}

export type RecoveryMode = 'smart' | 'literal_replay';
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
  /** Omitted inherits the source; null explicitly selects the bundled environment. */
  python_environment_path?: string | null;
}

export interface ResolvedModelInfo {
  provider: string;
  model: string;
  parameters: Record<string, unknown>;
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

/** Response of GET /api/settings. */
export interface ServerSettings {
  caribou_home: string;
  sessions_dir: string;
  uploads_dir: string;
  env_file: string;
  api_keys: Record<string, string>;
  ollama_host: string;
  ollama_model: string;
  /** null when no Slurm partition is configured on the server. */
  slurm_partition: string | null;
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
