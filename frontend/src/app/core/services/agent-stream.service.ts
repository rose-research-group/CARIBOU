import { Injectable, inject, signal, OnDestroy } from '@angular/core';
import { Subject, filter, Observable } from 'rxjs';
import {
  AgentEvent, AgentEventType, AgentEventEnvelope,
  TokenEventData, MessageCompleteData, AgentSwitchData,
  CodeSubmittedData, CodeResultData, ArtifactEventData,
  StatusChangeData, ErrorData, RecoveryProgressData, RecoveryCompletedData,
  SystemMessageData, WorkItemChangedData,
  BriefDraftData, BriefAcceptedData, PhaseChangeData
} from '../models/events.model';
import { SessionService } from './session.service';
import { SessionStatus } from '../models/session.model';

// Derive WebSocket base from document.baseURI so OOD path prefix is included.
// e.g. https://isvood001.mskcc.org/node/iscb014/8000/ → wss://isvood001.mskcc.org/node/iscb014/8000
const _base = document.baseURI.replace(/\/$/, '');
const WS_BASE = _base.replace(/^http/, 'ws');
const MAX_RETRIES = 5;
// Batch UI updates for token bursts. 50ms is imperceptible latency but
// collapses 20+ tiny signal writes into one per frame under heavy load.
const TOKEN_FLUSH_MS = 50;

export type WsConnectionState =
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'
  | 'expired';   // 4004 — session gone on server, no point retrying

@Injectable({ providedIn: 'root' })
export class AgentStreamService implements OnDestroy {
  private sessionSvc = inject(SessionService);

  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private retries = 0;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private tokenBuffer = '';
  private tokenFlushTimer: ReturnType<typeof setTimeout> | null = null;

  // Highest event seq applied for `seqSessionId`; survives reconnects so the
  // server's full-log replay on connect is not re-applied.
  private seqSessionId: string | null = null;
  private _lastSeq = signal(0);
  readonly lastSeq = this._lastSeq.asReadonly();

  private _events$ = new Subject<AgentEvent>();
  readonly events$: Observable<AgentEvent> = this._events$.asObservable();

  // Typed sub-streams
  readonly tokens$ = this.ofType<TokenEventData>('token');
  readonly messageComplete$ = this.ofType<MessageCompleteData>('message_complete');
  readonly agentSwitch$ = this.ofType<AgentSwitchData>('agent_switch');
  readonly codeSubmitted$ = this.ofType<CodeSubmittedData>('code_submitted');
  readonly codeResult$ = this.ofType<CodeResultData>('code_result');
  readonly artifacts$ = this.ofType<ArtifactEventData>('artifact');
  readonly statusChanges$ = this.ofType<StatusChangeData>('status_change');
  readonly recoveryProgress$ = this.ofType<RecoveryProgressData>('recovery_progress');
  readonly recoveryCompleted$ = this.ofType<RecoveryCompletedData>('recovery_completed');
  readonly systemMessages$ = this.ofType<SystemMessageData>('system_message');
  readonly workItemChanges$ = this.ofType<WorkItemChangedData>('work_item_changed');
  readonly briefDraft$ = this.ofType<BriefDraftData>('brief_draft');
  readonly briefAccepted$ = this.ofType<BriefAcceptedData>('brief_accepted');
  readonly phaseChange$ = this.ofType<PhaseChangeData>('phase_change');
  readonly errors$ = this.ofType<ErrorData>('error');

  // Current streaming state
  readonly streamingContent = signal<string>('');
  readonly streamingAgent = signal<string>('');
  readonly isStreaming = signal<boolean>(false);

  // Connection status observable so UI can render reconnecting banners.
  readonly connectionState = signal<WsConnectionState>('closed');
  readonly nextRetryAt = signal<number | null>(null);

  /**
   * Open the stream for `sessionId`. Reconnecting to the session already
   * tracked keeps the seq high-water mark, so the server's full-log replay is
   * not re-applied; a different session starts from 0.
   */
  connect(sessionId: string): void {
    const keepSeq = this.seqSessionId === sessionId ? this._lastSeq() : 0;
    this._teardown();
    this.sessionId = sessionId;
    this.seqSessionId = sessionId;
    this._lastSeq.set(keepSeq);
    this.retries = 0;
    this.connectionState.set('connecting');
    this._doConnect();
  }

  disconnect(): void {
    this._teardown();
    this.seqSessionId = null;
    this._lastSeq.set(0);
  }

  private _teardown(): void {
    this._clearPing();
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this._flushTokens();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.sessionId = null;
    this.streamingContent.set('');
    this.streamingAgent.set('');
    this.isStreaming.set(false);
    this.connectionState.set('closed');
    this.nextRetryAt.set(null);
  }

  send(msg: { type: string; content?: string }): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  startRun(prompt: string): void {
    this.send({ type: 'run', content: prompt });
  }

  sendUserMessage(content: string): void {
    this.send({ type: 'user_message', content });
  }

  stop(): void {
    this.send({ type: 'stop' });
  }

  cancelResponse(): void {
    this.send({ type: 'cancel_response' });
  }

  /** Force an immediate reconnect attempt (bypasses backoff). */
  retryNow(): void {
    if (!this.sessionId) return;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.retries = 0;
    this.connectionState.set('connecting');
    this.nextRetryAt.set(null);
    this._doConnect();
  }

  ngOnDestroy(): void {
    this.disconnect();
  }

  // ---------------------------------------------------------------------------

  private ofType<T>(type: AgentEventType): Observable<AgentEventEnvelope<T>> {
    return this.events$.pipe(
      filter(e => e.type === type)
    ) as Observable<AgentEventEnvelope<T>>;
  }

  private _doConnect(): void {
    if (!this.sessionId) return;
    const url = `${WS_BASE}/ws/sessions/${this.sessionId}`;
    this.ws = new WebSocket(url);

    this.ws.onmessage = (ev) => {
      // No catch: a malformed frame or a handler error must surface as an
      // uncaught error (reported by the global error listeners), not vanish.
      const frame = JSON.parse(ev.data) as { type: string; seq?: unknown };
      if (frame.type === 'pong') return;   // keepalive reply, not a session event
      if (typeof frame.seq !== 'number' || !Number.isInteger(frame.seq)) {
        throw new Error(`WebSocket event '${frame.type}' has no integer seq`);
      }
      const event = frame as AgentEvent;
      // The server replays the full event log on every (re)connect; every
      // event at or below the high-water mark has already been applied.
      if (event.seq <= this._lastSeq()) return;
      this._lastSeq.set(event.seq);
      this._handleEvent(event);
      this._events$.next(event);
    };

    this.ws.onopen = () => {
      this.retries = 0;
      this.connectionState.set('open');
      this.nextRetryAt.set(null);
      this._startPing();
    };

    this.ws.onclose = (ev) => {
      this._clearPing();
      // 4004 = session not found on server (e.g. after a server restart); no point retrying
      if (ev.code === 4004) {
        this.connectionState.set('expired');
        const session = this.sessionSvc.currentSession();
        if (session) {
          this.sessionSvc.updateLocal({ ...session, status: 'error' as SessionStatus });
        }
        return;
      }
      const session = this.sessionSvc.currentSession();
      const dead = session?.status === 'stopped' || session?.status === 'error';
      if (!dead && this.retries < MAX_RETRIES) {
        const delay = Math.min(1000 * 2 ** this.retries, 16000);
        this.retries++;
        this.connectionState.set('reconnecting');
        this.nextRetryAt.set(Date.now() + delay);
        this.reconnectTimer = setTimeout(() => this._doConnect(), delay);
      } else {
        this.connectionState.set('closed');
        this.nextRetryAt.set(null);
      }
    };

    this.ws.onerror = () => { this.ws?.close(); };
  }

  private _handleEvent(event: AgentEvent): void {
    switch (event.type) {
      case 'token': {
        const d = event.data as TokenEventData;
        this.isStreaming.set(true);
        if (this.streamingAgent() !== d.agent_name) {
          this.streamingAgent.set(d.agent_name);
        }
        this.tokenBuffer += d.token;
        this._scheduleTokenFlush();
        break;
      }
      case 'message_complete': {
        this._flushTokens();
        const d = event.data as MessageCompleteData;
        const session = this.sessionSvc.currentSession();
        if (session) {
          this.sessionSvc.updateLocal({
            ...session,
            current_turn: Math.max(session.current_turn, d.message.turn),
            message_count: session.message_count + 1,
          });
        }
        this.streamingContent.set('');
        this.streamingAgent.set('');
        this.isStreaming.set(false);
        break;
      }
      case 'status_change': {
        const d = event.data as StatusChangeData;
        const session = this.sessionSvc.currentSession();
        if (session) {
          this.sessionSvc.updateLocal({ ...session, status: d.status as SessionStatus });
        }
        break;
      }
      case 'recovery_progress': {
        const d = event.data as RecoveryProgressData;
        const session = this.sessionSvc.currentSession();
        if (session) {
          this.sessionSvc.updateLocal({
            ...session,
            recovery_phase: d.phase,
            recovery_detail: d.detail,
            recovery_step: d.step,
            recovery_total_steps: d.total_steps,
            recovery_substep: d.substep,
            recovery_substep_total: d.substep_total,
          });
        }
        break;
      }
      case 'recovery_completed': {
        const d = event.data as RecoveryCompletedData;
        const session = this.sessionSvc.currentSession();
        if (session) {
          this.sessionSvc.updateLocal({
            ...session,
            recovery_status: d.accepted_partial ? 'accepted_partial' : 'recovered',
            recovery_phase: 'completed',
            recovery_step: session.recovery_total_steps,
          });
        }
        break;
      }
      case 'agent_switch': {
        const d = event.data as AgentSwitchData;
        const session = this.sessionSvc.currentSession();
        if (session) {
          this.sessionSvc.updateLocal({ ...session, current_agent: d.to_agent });
        }
        break;
      }
      case 'error': {
        const d = event.data as ErrorData;
        if (d.fatal) {
          const session = this.sessionSvc.currentSession();
          if (session) {
            this.sessionSvc.updateLocal({ ...session, status: 'error' as SessionStatus });
          }
        }
        break;
      }
    }
  }

  private _scheduleTokenFlush(): void {
    if (this.tokenFlushTimer !== null) return;
    this.tokenFlushTimer = setTimeout(() => {
      this.tokenFlushTimer = null;
      this._flushTokens();
    }, TOKEN_FLUSH_MS);
  }

  private _flushTokens(): void {
    if (this.tokenFlushTimer !== null) {
      clearTimeout(this.tokenFlushTimer);
      this.tokenFlushTimer = null;
    }
    if (!this.tokenBuffer) return;
    const batch = this.tokenBuffer;
    this.tokenBuffer = '';
    this.streamingContent.update(c => c + batch);
  }

  private _startPing(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 20000);
  }

  private _clearPing(): void {
    if (this.pingInterval !== null) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
}
