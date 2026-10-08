export interface MachineObject {
  type: string;
  id: string;
  state: string;
}

export interface MachineResponse<T> {
  schema_version: 'caribou.machine_response.v1';
  command: string;
  ok: true;
  timestamp: string;
  caribou: { version: string; commit: string };
  object: MachineObject;
  data: T;
  links: Record<string, string>;
}

export interface CodeIdentity {
  repository: string;
  branch: string;
  commit: string;
  dirty: boolean;
}

export interface ModelIdentity {
  provider: string;
  model: string;
}

export interface BlueprintIdentity {
  topology: string;
  driver_agent: string;
  source: { uri: string; content_hash: string };
}

export interface ContainerIdentity {
  sandbox: string;
  image: { uri: string; content_hash: string };
  network_enabled: boolean;
}

export interface RunRecord {
  schema_version: 'caribou.run.v1';
  run_id: string;
  experiment_id: string;
  spec_hash: string;
  condition_id: string;
  replicate_index: number;
  attempt_index: number;
  interface: 'cli' | 'web' | 'benchmark' | 'migration';
  owner: string;
  state: string;
  executor: 'local' | 'slurm';
  code: CodeIdentity;
  resolved_model: ModelIdentity;
  resolved_blueprint: BlueprintIdentity;
  container: ContainerIdentity;
  scheduler_job_id: string | null;
  partition: string | null;
  resumed_from_run_id: string | null;
  resume_checkpoint_id: string | null;
  current_turn: number;
  current_agent: string | null;
  event_sequence: number;
  created_at: string;
  updated_at: string;
  terminal_outcome: string | null;
  end_reason: string | null;
  resume_eligible: boolean;
}

export interface ControlEvent {
  schema_version: 'caribou.event.v1';
  event_id: string;
  sequence: number;
  occurred_at: string;
  event_type: string;
  turn: number;
  stage: string | null;
  actor: string;
  payload: Record<string, unknown>;
}

export interface ControlArtifact {
  schema_version: 'caribou.artifact.v1';
  artifact_id: string;
  run_id: string;
  artifact_type: string;
  role: string;
  filename: string;
  content_hash: string;
  media_type: string;
  size_bytes: number;
  created_at: string;
  sensitivity: string;
}

export interface ControlCheckpoint {
  schema_version: 'caribou.checkpoint.v1';
  checkpoint_id: string;
  run_id: string;
  turn: number;
  stage: string;
  created_at: string;
  status: string;
}

export interface SubmitData {
  experiment: Record<string, unknown>;
  runs: RunRecord[];
  run_ids: string[];
  plan_hash: string;
  idempotent_replay: boolean;
  workers_launched: number;
}

export type PresetProfile = 'fast' | 'thorough';
export type PresetExecutor = 'local' | 'slurm';
export type PresetProvider = 'openai' | 'deepseek' | 'openrouter';

export const DEEPSEEK_MODELS = [
  {
    id: 'deepseek-v4-flash',
    label: 'DeepSeek V4 Flash (Quick)',
    thinking: false,
  },
  {
    id: 'deepseek-flash',
    label: 'DeepSeek V4.1 Flash (Quick)',
    thinking: false,
  },
  {
    id: 'deepseek-v4-pro',
    label: 'DeepSeek V4 Pro (Thinking)',
    thinking: true,
  },
] as const;

export interface PresetResourceProfile {
  cpu_cores: number;
  memory_bytes: number;
  wall_seconds: number;
}

export interface PresetSummary {
  id: string;
  name: string;
  description: string;
  default_profile: PresetProfile;
  default_max_turns: number;
  maximum_max_turns: number;
  resource_profiles: Record<PresetProfile, PresetResourceProfile>;
}

export interface PresetResolveRequest {
  dataset_path: string;
  model_provider: PresetProvider;
  model_name: string;
  openrouter_endpoint?: string;
  profile: PresetProfile;
  max_turns: number;
  executor: PresetExecutor;
  owner: string;
  reviewer: string;
  evaluator_provider?: PresetProvider;
  evaluator_model_name?: string;
}

export interface PresetResolveData {
  preset_id: string;
  spec_hash: string;
  specification: Record<string, unknown>;
  checks: Record<string, unknown>[];
}

export interface StatusData {
  run: RunRecord;
  cursor: number;
}

export interface EventsData {
  events: ControlEvent[];
  after: number;
  next_cursor: number;
  current_cursor: number;
  has_more: boolean;
}

export interface ArtifactsData {
  artifacts: ControlArtifact[];
  count: number;
}

export interface CheckpointsData {
  checkpoints: ControlCheckpoint[];
  count: number;
}

export interface VerifyData {
  verified: number;
  artifact_ids: string[];
}

export interface CancelData {
  run: RunRecord;
  applied: boolean;
  scheduler_signalled: boolean;
}

export interface CheckpointData {
  run: RunRecord;
  request: Record<string, unknown>;
  applied: boolean;
  safe_boundary: string;
}

export interface ResumeData {
  source_run: RunRecord;
  checkpoint: ControlCheckpoint;
  child_run: RunRecord;
  idempotent_replay: boolean;
  workers_launched: number;
}
