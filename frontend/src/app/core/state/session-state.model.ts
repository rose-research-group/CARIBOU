import { Message } from '../models/session.model';
import {
  CodeSubmittedData, CodeResultData, ErrorData, RecoveryCompletedData,
} from '../models/events.model';

export interface ErrorRecord {
  code: string;
  message: string;
  fatal: boolean;
  timestamp: string;
  suggested_fix?: string | null;
}

export interface StatusEntry {
  status: string;
  reason: string | null;
  timestamp: string;
  count: number;
}

export interface CodeEvent {
  submitted: CodeSubmittedData;
  result?: CodeResultData;
}

export interface ChatItem {
  kind: 'message' | 'delegation' | 'code' | 'error' | 'recovery';
  turn?: number;
  message?: Message;
  /** An `agent_switch`: a delegation, or a return to a work item's owner (`reason` says why). */
  delegation?: { from: string; to: string; command: string; reason?: string | null };
  codeEvent?: CodeEvent;
  error?: ErrorRecord;
  recovery?: RecoveryCompletedData & { id: string; timestamp: string };
}

/**
 * What applying one event asks of the page, for side effects that need
 * component APIs. The store has already updated its own state.
 */
export interface ApplyOutcome {
  /** Scroll the chat to the bottom, if the user has auto-scroll enabled. */
  scroll: boolean;
  /** A newly recorded non-fatal error, for the warning toast. */
  nonFatalError: ErrorData | null;
  /** The status from a `status_change` event (memory polling follows it). */
  status: string | null;
}
