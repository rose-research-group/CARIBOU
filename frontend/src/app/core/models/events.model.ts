import { Artifact, Message, WorkItemDetail, SessionBriefFields } from './session.model';

export interface AgentEventEnvelope<T = unknown> {
  type: AgentEventType;
  session_id: string;
  turn: number;
  timestamp: string;
  data: T;
  /** Server-assigned, strictly increasing per session (starts at 1); present on replay too. */
  seq: number;
}

export type AgentEventType =
  | 'token'
  | 'message_complete'
  | 'agent_switch'
  | 'code_submitted'
  | 'code_result'
  | 'artifact'
  | 'status_change'
  | 'metrics_result'
  | 'recovery_progress'
  | 'recovery_completed'
  | 'system_message'
  | 'work_item_changed'
  | 'brief_draft'
  | 'brief_accepted'
  | 'phase_change'
  | 'error'
  | 'pong';

export interface TokenEventData {
  agent_name: string;
  token: string;
}

export interface MessageCompleteData {
  message: Message;
}

export interface AgentSwitchData {
  from_agent: string;
  to_agent: string;
  command: string;
  reason: string | null;
}

export interface CodeSubmittedData {
  agent_name: string;
  source: string;
  block_index: number;
  total_blocks: number;
}

export interface CodeResultData {
  agent_name: string;
  stdout: string;
  stderr: string;
  success: boolean;
  duration_ms: number;
  block_index: number;
}

export interface ArtifactEventData {
  artifact: Artifact & { local_path: string };
}

export interface StatusChangeData {
  status: string;
  reason: string | null;
}

export interface ErrorData {
  code: string;
  message: string;
  fatal: boolean;
  suggested_fix?: string | null;
}

export interface RecoveryProgressData {
  phase: string;
  detail: string;
  step: number;
  total_steps: number;
  substep: number | null;
  substep_total: number | null;
  mode: string | null;
  attempt_number: number;
}

export interface RecoveryCompletedData {
  mode: string;
  attempt_number: number;
  checkpoint_id: string | null;
  checkpoint_turn: number | null;
  detail: string | null;
  accepted_partial: boolean;
}

export interface SystemMessageData {
  id: string;
  content: string;
  category: string;
}

export interface WorkItemChangedData {
  item: WorkItemDetail;
}

export interface BriefDraftData {
  brief: SessionBriefFields;
}

export interface BriefAcceptedData {
  brief: SessionBriefFields;
}

export interface PhaseChangeData {
  phase: 'briefing' | 'execution';
}

export type AgentEvent =
  | AgentEventEnvelope<TokenEventData>
  | AgentEventEnvelope<MessageCompleteData>
  | AgentEventEnvelope<AgentSwitchData>
  | AgentEventEnvelope<CodeSubmittedData>
  | AgentEventEnvelope<CodeResultData>
  | AgentEventEnvelope<ArtifactEventData>
  | AgentEventEnvelope<StatusChangeData>
  | AgentEventEnvelope<RecoveryProgressData>
  | AgentEventEnvelope<RecoveryCompletedData>
  | AgentEventEnvelope<SystemMessageData>
  | AgentEventEnvelope<WorkItemChangedData>
  | AgentEventEnvelope<BriefDraftData>
  | AgentEventEnvelope<BriefAcceptedData>
  | AgentEventEnvelope<PhaseChangeData>
  | AgentEventEnvelope<ErrorData>
  | AgentEventEnvelope<{}>;
