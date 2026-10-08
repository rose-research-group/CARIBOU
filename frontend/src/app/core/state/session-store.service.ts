import { Injectable, computed, effect, inject, signal, untracked } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { SessionService } from '../services/session.service';
import { AgentStreamService } from '../services/agent-stream.service';
import { SessionCacheService } from '../services/session-cache.service';
import { dedupeArtifactsByPath } from '../utils/artifacts';
import {
  Artifact, Message, QueuedMessage, SessionBriefFields, WorkItemDetail, WorkItemSummary,
} from '../models/session.model';
import { Block } from '../models/block.model';
import {
  AgentEvent, AgentEventEnvelope, AgentSwitchData, BlockChangedData, BriefDraftData,
  CodeResultData, CodeSubmittedData, ErrorData, MessageCompleteData, MessageQueueChangedData,
  RecoveryCompletedData, StatusChangeData, SystemMessageData, WorkItemChangedData,
} from '../models/events.model';
import { ApplyOutcome, ChatItem, ErrorRecord, StatusEntry } from './session-state.model';
import {
  addRecovery, appendCodeSubmitted, appendDelegation, appendStatus, attachCodeResult,
  errorRecordFrom, mergeBlockSnapshot, mergeServerMessages, pendingUserMessage,
  prunePendingRemovals, replaceMessageQueue, systemMessageFrom, upsertBlock, upsertMessage,
  upsertWorkItem, withPendingCode, withoutPendingCode,
} from './reducers';
import {
  ActivityMark, NO_ACTIVITY, parseServerTime, reduceActivityMark,
} from '../../pages/session/workbench/activity';

const CACHE_STALE_MS = 5 * 60 * 1000;

/**
 * Event-derived state of one session page. Provided by the session page
 * component (not root), so every page gets a fresh store. The page forwards
 * every WebSocket event to `apply`, which updates the signals below and
 * returns the side effects that need component APIs (scroll, toasts, polling).
 */
@Injectable()
export class SessionStore {
  private sessionSvc = inject(SessionService);
  private stream = inject(AgentStreamService);
  private cache = inject(SessionCacheService);

  private sessionId: string | null = null;
  private cacheHydrated = signal(false);
  // Highest event seq whose effects are already in the state restored from
  // sessionStorage. The server replays its full event log on connect, so
  // handlers that APPEND to cached state skip events at or below it.
  // (Reconnect replay within this page is dropped earlier, by the stream's own
  // seq high-water mark.) Idempotent handlers still run for those events so
  // un-cached state (pending code, brief draft) is rebuilt.
  private hydratedSeq = 0;

  private _chatItems = signal<ChatItem[]>([]);
  private _artifacts = signal<Artifact[]>([]);
  private _pendingCode = signal<Map<string, CodeSubmittedData>>(new Map());
  private _errorLog = signal<ErrorRecord[]>([]);
  private _statusLog = signal<StatusEntry[]>([]);
  private _workItems = signal<WorkItemSummary[]>([]);
  private _selectedWorkItem = signal<WorkItemDetail | null>(null);
  private _waitingForAgent = signal(false);
  private _cancellingResponse = signal(false);
  private _awaitingCodeResult = signal(false);
  // WS-5: the briefing conversation is a distinct phase, not a normal chat
  // turn — `briefDraft` is the agent's current proposal (cleared once a
  // decision is submitted).
  private _briefDraft = signal<SessionBriefFields | null>(null);
  private _briefEditDraft = signal('');
  private _briefEditing = signal(false);
  private _submittingBriefDecision = signal(false);
  private _blocks = signal<Block[]>([]);
  private _blocksRecorded = signal<boolean | null>(null);
  private _blocksError = signal<string | null>(null);
  // Full work items (with reviews) for the workbench, keyed by id. Kept apart
  // from `workItems` (REST summaries) and `selectedWorkItem` (chat sidebar).
  private _workItemDetails = signal<Map<number, WorkItemDetail>>(new Map());
  private _workItemDetailErrors = signal<Map<number, string>>(new Map());
  private workItemDetailsRequested = new Set<number>();
  // Messages the server holds until the agent is ready (`message_queue_changed`
  // is the source of truth), and the ids whose removal was asked for but not
  // yet confirmed by one.
  private _messageQueue = signal<QueuedMessage[]>([]);
  private _pendingRemovals = signal<Set<string>>(new Set());
  // The workbench's working indicator: the latest notable event of the turn,
  // when the agent started working (ms), and the block a message sent from
  // this page focuses (null for chat messages and once the agent is idle).
  private _activityMark = signal<ActivityMark>(NO_ACTIVITY);
  private _workingSince = signal<number | null>(null);
  private _inFlightBlockId = signal<string | null>(null);

  readonly chatItems = this._chatItems.asReadonly();
  readonly artifacts = this._artifacts.asReadonly();
  /** Submitted code awaiting its result, keyed by `codeKey(turn, block_index)`. */
  readonly pendingCode = this._pendingCode.asReadonly();
  readonly errorLog = this._errorLog.asReadonly();
  readonly statusLog = this._statusLog.asReadonly();
  readonly workItems = this._workItems.asReadonly();
  readonly selectedWorkItem = this._selectedWorkItem.asReadonly();
  readonly waitingForAgent = this._waitingForAgent.asReadonly();
  readonly cancellingResponse = this._cancellingResponse.asReadonly();
  readonly awaitingCodeResult = this._awaitingCodeResult.asReadonly();
  readonly briefDraft = this._briefDraft.asReadonly();
  readonly briefEditDraft = this._briefEditDraft.asReadonly();
  readonly briefEditing = this._briefEditing.asReadonly();
  readonly submittingBriefDecision = this._submittingBriefDecision.asReadonly();
  /** One entry per block_id, ordered by `index`. */
  readonly blocks = this._blocks.asReadonly();
  /** From `GET .../blocks`: false for sessions with no blocks.json; null until known. */
  readonly blocksRecorded = this._blocksRecorded.asReadonly();
  /** Why `GET .../blocks` failed, if it did. */
  readonly blocksError = this._blocksError.asReadonly();
  readonly workItemDetails = this._workItemDetails.asReadonly();
  readonly workItemDetailErrors = this._workItemDetailErrors.asReadonly();
  /** Queued user messages, in delivery order. */
  readonly messageQueue = this._messageQueue.asReadonly();
  /** Queued message ids with a removal request in flight. */
  readonly pendingRemovals = this._pendingRemovals.asReadonly();
  readonly activityMark = this._activityMark.asReadonly();
  readonly workingSince = this._workingSince.asReadonly();
  readonly inFlightBlockId = this._inFlightBlockId.asReadonly();
  /** The accepted (frozen) brief: on the Session record once briefing is over. */
  readonly frozenBrief = computed(() => {
    const s = this.sessionSvc.currentSession();
    return s && s.phase !== 'briefing' ? s.brief : null;
  });

  readonly lastError = computed(() => {
    const errs = this._errorLog();
    return errs.length ? errs[errs.length - 1] : null;
  });
  readonly hasRecoveryMilestone = computed(() =>
    this._chatItems().some(item => item.kind === 'recovery')
  );

  constructor() {
    // Persist chat / artifacts / logs to sessionStorage for refresh recovery.
    effect(() => {
      const s = this.sessionSvc.currentSession();
      if (!s || !this.cacheHydrated()) return;
      this.cache.write(s.id, {
        chatItems: this._chatItems(),
        artifacts: this._artifacts(),
        errorLog: this._errorLog(),
        statusLog: this._statusLog(),
        // untracked: every event (tokens included) bumps seq; only changes to
        // the cached signals above should trigger a rewrite.
        lastSeq: Math.max(this.hydratedSeq, untracked(() => this.stream.lastSeq())),
      });
    });
  }

  /**
   * Bind the store to `sessionId` and restore its sessionStorage snapshot (if
   * fresh) so a refresh isn't jarring. Call once, before connecting the stream.
   */
  attach(sessionId: string): void {
    if (this.sessionId !== null) {
      throw new Error(`SessionStore is already attached to session ${this.sessionId}`);
    }
    this.sessionId = sessionId;
    const cached = this.cache.read(sessionId);
    if (cached) {
      const stale = Date.now() - cached.cachedAt > CACHE_STALE_MS;
      if (!stale) {
        this._chatItems.set(cached.chatItems as ChatItem[] ?? []);
        this._artifacts.set(cached.artifacts as Artifact[] ?? []);
        this._errorLog.set(cached.errorLog as ErrorRecord[] ?? []);
        this._statusLog.set(cached.statusLog as StatusEntry[] ?? []);
        this.hydratedSeq = cached.lastSeq;
      }
    }
    this.cacheHydrated.set(true);
  }

  /** The single reducer for WebSocket events. */
  apply(event: AgentEvent): ApplyOutcome {
    const outcome: ApplyOutcome = { scroll: false, nonFatalError: null, status: null };
    const replayed = event.seq <= this.hydratedSeq;
    this._activityMark.update(mark => reduceActivityMark(mark, event));
    switch (event.type) {
      case 'message_complete': {
        const d = event.data as MessageCompleteData;
        this._chatItems.update(items => upsertMessage(items, d.message));
        this._awaitingCodeResult.set(false);
        outcome.scroll = true;
        break;
      }
      case 'agent_switch': {
        if (replayed) break;
        const ev = event as AgentEventEnvelope<AgentSwitchData>;
        this._chatItems.update(items => appendDelegation(items, ev));
        break;
      }
      case 'code_submitted': {
        const ev = event as AgentEventEnvelope<CodeSubmittedData>;
        this._pendingCode.update(m => withPendingCode(m, ev.turn, ev.data));
        this._awaitingCodeResult.set(true);
        // Card already restored from the session cache.
        if (replayed) break;
        this._chatItems.update(items => appendCodeSubmitted(items, ev));
        outcome.scroll = true;
        break;
      }
      case 'code_result': {
        const result = event.data as CodeResultData;
        this._pendingCode.update(m => withoutPendingCode(m, event.turn, result.block_index));
        const next = attachCodeResult(this._chatItems(), event.turn, result);
        this._chatItems.set(next.items);
        outcome.scroll = next.attached;
        if (this._pendingCode().size === 0) {
          this._awaitingCodeResult.set(false);
        }
        break;
      }
      case 'artifact': {
        // The event payload carries no artifact id (needed for the download
        // URL), so refetch the list; an overwritten file keeps its `path` and
        // replaces the existing entry rather than adding a duplicate card.
        this.refreshArtifacts();
        break;
      }
      case 'work_item_changed': {
        const d = event.data as WorkItemChangedData;
        this._workItems.update(items => upsertWorkItem(items, d.item));
        if (this._selectedWorkItem()?.id === d.item.id) {
          this._selectedWorkItem.set(d.item);
        }
        this._workItemDetails.update(m => new Map(m).set(d.item.id, d.item));
        break;
      }
      case 'brief_draft': {
        const d = event.data as BriefDraftData;
        this._briefDraft.set(d.brief);
        this._briefEditDraft.set(JSON.stringify(d.brief, null, 2));
        this._briefEditing.set(false);
        break;
      }
      case 'brief_accepted': {
        this._briefDraft.set(null);
        this._briefEditing.set(false);
        this._submittingBriefDecision.set(false);
        break;
      }
      case 'phase_change': {
        // Phase lives on the Session record itself, not just the event
        // stream — refetch so `briefPhase`/`frozenBrief` pick up the change.
        this.sessionSvc.getSession(this.requireSessionId()).subscribe();
        break;
      }
      case 'error': {
        this._waitingForAgent.set(false);
        // A removal the server refused (e.g. QUEUED_MESSAGE_GONE) is settled;
        // the error itself is recorded below like any other.
        this._pendingRemovals.set(new Set());
        this._inFlightBlockId.set(null);
        if (replayed) break;
        const ev = event as AgentEventEnvelope<ErrorData>;
        const record = errorRecordFrom(ev);
        this._errorLog.update(errs => [...errs, record]);
        this._chatItems.update(items => [...items, { kind: 'error', turn: ev.turn, error: record }]);
        outcome.scroll = true;
        if (!ev.data.fatal) outcome.nonFatalError = ev.data;
        break;
      }
      case 'status_change': {
        const d = event.data as StatusChangeData;
        if (d.status === 'idle' || d.status === 'stopped' || d.status === 'error') {
          this._waitingForAgent.set(false);
          this._cancellingResponse.set(false);
          this._workingSince.set(null);
          this._inFlightBlockId.set(null);
        } else if (this._workingSince() === null &&
            (d.status === 'running' || d.status === 'initializing' || d.status === 'recovering')) {
          this._workingSince.set(parseServerTime(event.timestamp));
        }
        outcome.status = d.status;
        if (replayed) break;
        this._statusLog.update(log => appendStatus(log, d, event.timestamp));
        break;
      }
      case 'system_message': {
        const ev = event as AgentEventEnvelope<SystemMessageData>;
        this._chatItems.update(items => upsertMessage(items, systemMessageFrom(ev)));
        break;
      }
      case 'recovery_completed': {
        const ev = event as AgentEventEnvelope<RecoveryCompletedData>;
        this._chatItems.update(items => addRecovery(items, ev));
        outcome.scroll = true;
        break;
      }
      case 'block_changed': {
        const d = event.data as BlockChangedData;
        this._blocks.update(blocks => upsertBlock(blocks, d.block));
        this._blocksRecorded.set(true);
        break;
      }
      case 'message_queue_changed': {
        const d = event.data as MessageQueueChangedData;
        this.setMessageQueue(d.queue);
        break;
      }
      // token / recovery_progress are reduced by AgentStreamService and the
      // Session record; metrics_result and pong carry no page state.
    }
    return outcome;
  }

  /**
   * Replace chat messages with the server's list, but only when the server
   * has at least as many as the chat (else the cache is newer). Returns
   * whether the chat was replaced.
   */
  mergeServerMessages(msgs: Message[]): boolean {
    const merged = mergeServerMessages(this._chatItems(), msgs);
    if (merged === null) return false;
    this._chatItems.set(merged);
    return true;
  }

  /**
   * The queue from the session record on load. Call it before the stream
   * connects: replayed `message_queue_changed` events then supersede it.
   */
  hydrateMessageQueue(queue: QueuedMessage[]): void {
    if (!Array.isArray(queue)) {
      throw new Error('The session record has no message_queue list.');
    }
    this.setMessageQueue(queue);
  }

  /** Mark a queued message's removal as sent; the server's next queue settles it. */
  markRemovalPending(id: string): void {
    this._pendingRemovals.update(ids => new Set(ids).add(id));
  }

  private setMessageQueue(queue: QueuedMessage[]): void {
    this._messageQueue.update(current => replaceMessageQueue(current, queue));
    this._pendingRemovals.update(ids => prunePendingRemovals(ids, queue));
  }

  refreshArtifacts(): void {
    this.sessionSvc.getArtifacts(this.requireSessionId())
      .subscribe(a => this._artifacts.set(dedupeArtifactsByPath(a)));
  }

  /** Load `GET /api/sessions/{id}/blocks`, merged with blocks seen live. */
  hydrateBlocks(): void {
    this.sessionSvc.getBlocks(this.requireSessionId()).subscribe({
      next: res => {
        // Never downgrade: replayed block_changed events may have arrived
        // before a response read while blocks.json did not exist yet.
        this._blocksRecorded.set(res.recorded || this._blocks().length > 0);
        this._blocks.update(blocks => mergeBlockSnapshot(blocks, res.blocks));
      },
      error: (err: HttpErrorResponse) => this._blocksError.set(err.message),
    });
  }

  /**
   * Fetch one work item with its reviews into `workItemDetails`, once per id
   * (later changes arrive through `work_item_changed`). A failure is kept in
   * `workItemDetailErrors`.
   */
  loadWorkItemDetail(itemId: number): void {
    if (this.workItemDetailsRequested.has(itemId) || this._workItemDetails().has(itemId)) return;
    this.workItemDetailsRequested.add(itemId);
    this.sessionSvc.getWorkItem(this.requireSessionId(), itemId).subscribe({
      // Never downgrade: a work_item_changed event that arrived while the
      // request was in flight is at least as new as this response.
      next: item => this._workItemDetails.update(m => m.has(item.id) ? m : new Map(m).set(item.id, item)),
      error: (err: HttpErrorResponse) =>
        this._workItemDetailErrors.update(m => new Map(m).set(itemId, err.message)),
    });
  }

  setWorkItems(items: WorkItemSummary[]): void {
    this._workItems.set(items);
  }

  upsertWorkItem(item: WorkItemSummary): void {
    this._workItems.update(items => upsertWorkItem(items, item));
  }

  setSelectedWorkItem(item: WorkItemDetail | null): void {
    this._selectedWorkItem.set(item);
  }

  /**
   * Show the user's message optimistically and wait for the agent. `blockId`
   * is the workbench block the message focuses, if any.
   */
  sendUserMessage(sessionId: string, turn: number, content: string, blockId: string | null = null): void {
    this._waitingForAgent.set(true);
    this._inFlightBlockId.set(blockId);
    if (this._workingSince() === null) this._workingSince.set(Date.now());
    this._chatItems.update(items => [...items, pendingUserMessage(sessionId, turn, content)]);
  }

  markCancelling(): void {
    this._cancellingResponse.set(true);
  }

  setSubmittingBriefDecision(submitting: boolean): void {
    this._submittingBriefDecision.set(submitting);
  }

  clearBriefDraft(): void {
    this._briefDraft.set(null);
  }

  setBriefEditDraft(text: string): void {
    this._briefEditDraft.set(text);
  }

  setBriefEditing(editing: boolean): void {
    this._briefEditing.set(editing);
  }

  private requireSessionId(): string {
    if (this.sessionId === null) {
      throw new Error('SessionStore used before attach(sessionId)');
    }
    return this.sessionId;
  }
}
