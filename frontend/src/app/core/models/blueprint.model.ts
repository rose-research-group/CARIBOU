export interface CommandConfig {
  target_agent: string;
  description: string;
}

export interface AgentConfig {
  prompt: string;
  rag_enabled: boolean;
  neighbors: Record<string, CommandConfig>;
  code_samples: string[];
}

export interface BlueprintContent {
  name: string;
  global_policy: string;
  agents: Record<string, AgentConfig>;
  is_package_default: boolean;
  evaluator_agent: string | null;
  work_item_policy: WorkItemPolicyConfig;
  // Opaque: same JSON shape as the blueprint file; round-tripped unchanged.
  brief_policy?: BriefPolicyConfig | null;
}

export type BriefPolicyConfig = Record<string, unknown>;

export interface WorkItemPolicyConfig {
  qc_mode: 'optional' | 'required';
}

export interface SaveBlueprintRequest {
  name: string;
  global_policy: string;
  agents: Record<string, AgentConfig>;
  evaluator_agent: string | null;
  work_item_policy: WorkItemPolicyConfig;
  // Opaque: same JSON shape as the blueprint file; round-tripped unchanged.
  brief_policy?: BriefPolicyConfig | null;
}

// Local editor state types (not sent over the wire)
export interface CommandEntry {
  key: string;
  target_agent: string;
  description: string;
}

export interface AgentEntry {
  key: string;
  prompt: string;
  ragEnabled: boolean;
  commands: CommandEntry[];
  codeSamples: string[];
}
