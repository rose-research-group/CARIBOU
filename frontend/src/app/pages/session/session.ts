import {
  Component, OnInit, OnDestroy, inject, signal, ViewChild,
  ElementRef, AfterViewChecked, computed, HostListener, effect
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { Subscription, interval } from 'rxjs';
import { SessionService } from '../../core/services/session.service';
import { ConfigService } from '../../core/services/config.service';
import { AgentStreamService } from '../../core/services/agent-stream.service';
import { ToastService } from '../../core/services/toast.service';
import { PreferencesService } from '../../core/services/preferences.service';
import { SessionCacheService } from '../../core/services/session-cache.service';
import {
  Message, Artifact, MemoryState, EvaluationResult, EvaluatorModelConfig,
  RecoveryMode, SessionForkRequest, SessionResumeRequest,
  WorkItemDetail, WorkItemSummary,
} from '../../core/models/session.model';
import {
  MessageCompleteData, AgentSwitchData, CodeSubmittedData,
  CodeResultData, ErrorData, StatusChangeData, RecoveryCompletedData,
  SystemMessageData, WorkItemChangedData,
} from '../../core/models/events.model';
import { MessageBubbleComponent } from '../../shared/components/message-bubble/message-bubble';
import { ArtifactCardComponent } from '../../shared/components/artifact-card/artifact-card';
import { StatusIndicatorComponent } from '../../shared/components/status-indicator/status-indicator';
import { IconComponent } from '../../shared/components/icon/icon';
import { TooltipDirective } from '../../shared/directives/tooltip.directive';
import { navigateTabToSession, reserveNewTab } from '../../core/utils/app-navigation';

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

interface ChatItem {
  kind: 'message' | 'delegation' | 'code' | 'error' | 'recovery';
  turn?: number;
  message?: Message;
  delegation?: { from: string; to: string; command: string };
  codeEvent?: { submitted: CodeSubmittedData; result?: CodeResultData };
  error?: ErrorRecord;
  recovery?: RecoveryCompletedData & { id: string; timestamp: string };
}

const COMPACT_AFTER_ITEMS = 40;
const VISIBLE_RECENT_ITEMS = 20;
const AUTO_SCROLL_THRESHOLD_PX = 120;
const INPUT_HISTORY_KEY = 'caribou:input-history:v1';
const INPUT_HISTORY_LIMIT = 25;

type ArtifactFilter = 'all' | 'plot' | 'data' | 'other';

@Component({
  selector: 'app-session',
  standalone: true,
  imports: [
    CommonModule, FormsModule,
    MessageBubbleComponent, ArtifactCardComponent, StatusIndicatorComponent,
    IconComponent, TooltipDirective,
  ],
  templateUrl: './session.html',
  styleUrl: './session.scss',
})
export class SessionComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('chatBottom') chatBottom!: ElementRef;
  @ViewChild('chatPanel') chatPanel!: ElementRef<HTMLElement>;
  @ViewChild('messageInput') messageInput!: ElementRef<HTMLTextAreaElement>;
  readonly COMPACT_AFTER_ITEMS = COMPACT_AFTER_ITEMS;

  private route = inject(ActivatedRoute);
  private router = inject(Router);
  sessionSvc = inject(SessionService);
  configSvc = inject(ConfigService);
  stream = inject(AgentStreamService);
  private toasts = inject(ToastService);
  private prefsSvc = inject(PreferencesService);
  private cache = inject(SessionCacheService);

  chatItems = signal<ChatItem[]>([]);
  olderConversationExpanded = signal(false);
  artifacts = signal<Artifact[]>([]);
  userInput = signal('');
  pendingCode = signal<Map<string, CodeSubmittedData>>(new Map());
  errorLog = signal<ErrorRecord[]>([]);
  statusLog = signal<StatusEntry[]>([]);
  waitingForAgent = signal(false);
  cancellingResponse = signal(false);
  awaitingCodeResult = signal(false);
  showTimeline = signal(false);
  artifactFilter = signal<ArtifactFilter>('all');
  artifactSearch = signal('');
  memoryState = signal<MemoryState | null>(null);
  memoryStateError = signal(false);
  evaluating = signal(false);
  evaluationResult = signal<EvaluationResult | null>(null);
  evaluationError = signal<string | null>(null);
  workItems = signal<WorkItemSummary[]>([]);
  selectedWorkItem = signal<WorkItemDetail | null>(null);
  workItemReviewing = signal(false);
  workItemError = signal<string | null>(null);
  editingEvaluatorModel = signal(false);
  evaluatorModelSaving = signal(false);
  evaluatorModelError = signal<string | null>(null);
  evaluatorModelReason = signal('');
  evaluatorModelForm: EvaluatorModelConfig = { mode: 'inherit_worker' };
  showConnectionBanner = computed(() => {
    const s = this.stream.connectionState();
    return s === 'reconnecting' || s === 'expired';
  });
  reconnectCountdownSec = signal(0);
  showShortcutHelp = signal(false);
  showContext = signal(false);
  recoveryDialog = signal<'resume' | 'fork' | 'retry' | null>(null);
  recoverySubmitting = signal(false);
  recoveryError = signal<string | null>(null);
  recoveryForm: SessionForkRequest = {
    name: '',
    recovery_mode: 'smart',
    target_mode: 'interactive',
    additional_turns: 20,
    acknowledge_replay_risk: false,
  };
  autoScrollEnabled = signal(true);
  showJumpToLatest = signal(false);
  sessionElapsedSec = signal(0);
  private sessionStartTs: number | null = null;
  private lastCompletedStatus: string | null = null;
  private originalTitle = typeof document !== 'undefined' ? document.title : '';
  private inputHistory: string[] = this.loadHistory();
  private historyIndex = -1;
  private historyDraft = '';

  private subs = new Subscription();
  private memoryPollSub: Subscription | null = null;
  private shouldScrollToBottom = false;
  private cacheHydrated = false;

  session = this.sessionSvc.currentSession;
  status = computed(() => this.session()?.status ?? 'stopped');
  currentAgent = computed(() => this.session()?.current_agent ?? '');
  isIdle = computed(() => this.status() === 'idle');
  isRunning = computed(() => this.status() === 'running');
  isStopped = computed(() => this.status() === 'stopped');
  isError = computed(() => this.status() === 'error');
  isInitializing = computed(() => this.status() === 'initializing');
  isRecovering = computed(() => this.status() === 'recovering');
  developerMode = computed(() => this.prefsSvc.prefs().developerMode);
  recoveryStages = [
    'Safe checkpoint',
    'Previous runtime',
    'Agent & model',
    'Fresh sandbox',
    'History & memory',
    'Dataset setup',
    'Environment rebuild',
    'Start session',
  ];
  recoveryPercent = computed(() => {
    const s = this.session();
    if (!s?.recovery_total_steps) return 4;
    return Math.max(4, Math.min(100, (s.recovery_step / s.recovery_total_steps) * 100));
  });
  isThinking = computed(() => this.isRunning() && !this.stream.isStreaming() && !this.awaitingCodeResult());
  isInitialInteractiveTurn = computed(() =>
    this.session()?.mode === 'interactive' && (this.session()?.current_turn ?? 0) <= 1
  );
  // Mirrors the server's own readiness check — false after a server restart
  // until the runner is relaunched, even if current_turn > 0 (see
  // can_evaluate in session_state.py's to_response()).
  canEvaluate = computed(() => this.session()?.can_evaluate ?? false);
  waitingLabel = computed(() =>
    this.isInitialInteractiveTurn() ? 'Getting environment set up…' : 'Waiting for agent…'
  );
  thinkingLabel = computed(() =>
    this.isInitialInteractiveTurn() ? 'Getting environment set up…' : (this.currentAgent() || 'Agent')
  );
  lastError = computed(() => {
    const errs = this.errorLog();
    return errs.length ? errs[errs.length - 1] : null;
  });
  runState = computed<'idle' | 'thinking' | 'generating' | 'executing' | 'stopped' | 'error' | 'initializing' | 'recovering'>(() => {
    if (this.isError()) return 'error';
    if (this.isInitializing()) return 'initializing';
    if (this.isRecovering()) return 'recovering';
    if (this.status() === 'stopped') return 'stopped';
    if (this.awaitingCodeResult()) return 'executing';
    if (this.stream.isStreaming()) return 'generating';
    if (this.isRunning()) return 'thinking';
    return 'idle';
  });
  runStateLabel = computed(() => {
    switch (this.runState()) {
      case 'generating':   return 'Agent generating';
      case 'thinking':     return 'Agent thinking';
      case 'executing':    return 'Running code';
      case 'initializing': return 'Initializing';
      case 'recovering':   return 'Recovering';
      case 'stopped':      return 'Stopped';
      case 'error':        return 'Errored';
      default:             return 'Idle';
    }
  });
  displayChatItems = computed(() =>
    this.chatItems().filter(item =>
      this.developerMode() || item.kind !== 'message' || item.message?.role !== 'system'
    )
  );
  hasRecoveryMilestone = computed(() =>
    this.chatItems().some(item => item.kind === 'recovery')
  );
  hiddenChatItemCount = computed(() => {
    const count = this.displayChatItems().length;
    return !this.olderConversationExpanded() && count > COMPACT_AFTER_ITEMS
      ? count - VISIBLE_RECENT_ITEMS
      : 0;
  });
  visibleChatItems = computed(() => {
    const items = this.displayChatItems();
    const hidden = this.hiddenChatItemCount();
    return hidden ? items.slice(hidden) : items;
  });
  /** Turn -> chatItem index (first item at that turn). Used for jump-to-turn. */
  turnAnchors = computed(() => {
    const anchors: { id: string; turn: number; label: string; selector: string }[] = [];
    const items = this.displayChatItems();
    const seen = new Set<number>();
    items.forEach(item => {
      const turn = item.turn ?? item.message?.turn ?? 0;
      if (item.kind === 'recovery' && item.recovery) {
        anchors.push({
          id: item.recovery.id,
          turn,
          label: `Recovery · ${this.recoveryModeLabel(item.recovery.mode)}`,
          selector: `[data-recovery-id="${item.recovery.id}"]`,
        });
        return;
      }
      if (turn && !seen.has(turn)) {
        seen.add(turn);
        anchors.push({ id: `turn-${turn}`, turn, label: `Turn ${turn}`, selector: `[data-turn="${turn}"]` });
      }
    });
    return anchors;
  });

  avgTurnDurationMs = computed(() => {
    const s = this.session();
    if (!s || !s.current_turn || !this.sessionStartTs) return 0;
    const elapsed = this.sessionElapsedSec() * 1000;
    if (elapsed <= 0) return 0;
    return elapsed / Math.max(s.current_turn, 1);
  });

  eta = computed(() => {
    const s = this.session();
    if (!s || !s.max_turns || !s.current_turn) return null;
    const avg = this.avgTurnDurationMs();
    if (avg <= 0) return null;
    const remaining = Math.max(0, s.max_turns - s.current_turn);
    if (!remaining) return null;
    return Math.round((remaining * avg) / 1000);
  });

  filteredArtifacts = computed(() => {
    const filter = this.artifactFilter();
    const q = this.artifactSearch().trim().toLowerCase();
    return this.artifacts()
      .filter(a => {
        if (filter === 'all') return true;
        if (filter === 'plot') return a.type === 'plot';
        if (filter === 'data') return a.type === 'data';
        return a.type !== 'plot' && a.type !== 'data';
      })
      .filter(a => !q || a.filename.toLowerCase().includes(q));
  });

  artifactCounts = computed(() => {
    const arts = this.artifacts();
    return {
      all: arts.length,
      plot: arts.filter(a => a.type === 'plot').length,
      data: arts.filter(a => a.type === 'data').length,
      other: arts.filter(a => a.type !== 'plot' && a.type !== 'data').length,
    };
  });

  memoryActive = computed(() => {
    const s = this.session();
    return s?.memory?.strategy && s.memory.strategy !== 'full';
  });

  contextBreakdown = computed(() => {
    const ms = this.memoryState();
    if (!ms?.context_breakdown) return null;
    const bd = ms.context_breakdown;
    // Bars are sized by estimated token share — tokens are what actually
    // determines context-window pressure, message count is just a detail.
    const maxTokens = Math.max(1, bd.total_tokens ?? bd.total);
    const seg = (label: string, msgs: number, tokens: number | undefined, color: string) => ({
      label, msgs, tokens: tokens ?? 0, pct: (tokens ?? 0) / maxTokens, color,
    });
    return [
      seg('Pinned system', bd.pinned_system, bd.pinned_system_tokens, 'var(--cividis-navy)'),
      seg('Pivotal code', bd.pivotal_code, bd.pivotal_code_tokens, 'var(--cividis-teal)'),
      seg('Summaries', bd.summaries, bd.summaries_tokens, 'var(--cividis-gold)'),
      seg('Working: user', bd.working_user, bd.working_user_tokens, '#6f42c1'),
      seg('Working: assistant', bd.working_assistant, bd.working_assistant_tokens, '#28a745'),
      seg('Working: system', bd.working_system, bd.working_system_tokens, 'var(--text-muted)'),
    ].filter(b => b.msgs > 0);
  });

  contextSummary = computed(() => {
    const ms = this.memoryState();
    if (!ms) return null;
    const bd = ms.context_breakdown;
    return {
      ...bd,
      strategy: ms.strategy,
      summarized_message_count: ms.summarized_message_count ?? 0,
      total_messages: ms.total_messages ?? bd.total_full_history ?? bd.total,
      total_tokens: bd.total_tokens ?? ms.context_estimate_tokens,
      total_full_history_tokens: bd.total_full_history_tokens ?? ms.total_full_history_tokens,
      working_history_size: ms.config?.['working_history_size'] ?? this.session()?.memory?.working_history_size,
      summarization_threshold: ms.config?.['summarization_threshold'] ?? this.session()?.memory?.summarization_threshold,
      chunk_size: ms.config?.['chunk_size_to_summarize'] ?? this.session()?.memory?.chunk_size,
    };
  });

  constructor() {
    // Reactive tab-title notifications when session completes.
    effect(() => {
      const status = this.status();
      const prefs = this.prefsSvc.prefs();
      const s = this.session();
      if (!s) return;
      // Only fire once per transition into stopped/error.
      if (status === 'stopped' || status === 'error') {
        if (this.lastCompletedStatus !== status) {
          this.lastCompletedStatus = status;
          this.notifyCompletion(status);
        }
        if (prefs.tabTitleNotifications && document.hidden) {
          const prefix = status === 'stopped' ? '(done)' : '(error)';
          document.title = `${prefix} ${this.originalTitle}`;
        }
      } else if (status === 'running' || status === 'initializing' || status === 'recovering' || status === 'idle') {
        this.lastCompletedStatus = null;
        document.title = this.originalTitle;
      }
    });

    // Persist chat / artifacts / logs to sessionStorage for refresh recovery.
    effect(() => {
      const s = this.session();
      if (!s || !this.cacheHydrated) return;
      this.cache.write(s.id, {
        chatItems: this.chatItems(),
        artifacts: this.artifacts(),
        errorLog: this.errorLog(),
        statusLog: this.statusLog(),
      });
    });
  }

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id')!;
    // Hydrate from sessionStorage FIRST so refresh isn't jarring.
    const cached = this.cache.read(id);
    if (cached) {
      const stale = Date.now() - cached.cachedAt > 5 * 60 * 1000;
      if (!stale) {
        this.chatItems.set(cached.chatItems as ChatItem[] ?? []);
        this.artifacts.set(cached.artifacts as Artifact[] ?? []);
        this.errorLog.set(cached.errorLog as ErrorRecord[] ?? []);
        this.statusLog.set(cached.statusLog as StatusEntry[] ?? []);
      }
    }
    this.cacheHydrated = true;

    this.sessionSvc.getSession(id).subscribe({
      next: () => {
        // Validate the REST resource before opening a socket. Otherwise a stale
        // deep link briefly enters the WebSocket "expired" state before routing
        // back to the dashboard.
        this.stream.connect(id);
        this.sessionSvc.getMessages(id).subscribe(msgs => {
          // Only replace chat when server has more/newer messages than cache.
          if (msgs.length >= this.chatItems().filter(c => c.kind === 'message').length) {
            const items: ChatItem[] = msgs.map(m => ({ kind: 'message' as const, message: m, turn: m.turn }));
            const eventItems = this.chatItems().filter(item => item.kind !== 'message');
            this.chatItems.set(
              [...items, ...eventItems].sort(
                (a, b) => (a.turn ?? a.message?.turn ?? 0) - (b.turn ?? b.message?.turn ?? 0)
              )
            );
            this.shouldScrollToBottom = true;
          }
        });
        this.sessionSvc.getArtifacts(id).subscribe(a => this.artifacts.set(a));
        this.loadWorkItems(id);
        this.sessionStartTs = Date.now();
        this.fetchMemoryState(id);
      },
      error: (error: HttpErrorResponse) => {
        if (error.status === 404) {
          this.stream.disconnect();
          this.sessionSvc.clearCurrentSession();
          this.toasts.show({
            kind: 'warn',
            title: 'Session not found',
            detail: 'That session no longer exists. Returning to the dashboard.',
            ttlMs: 6000,
          });
          void this.router.navigate(['/'], { replaceUrl: true });
          return;
        }
        this.toasts.show({
          kind: 'error',
          title: 'Failed to load session',
          detail: 'Could not fetch session details from the server.',
          ttlMs: 8000,
        });
      },
    });

    this.configSvc.loadAll().subscribe();

    // Sub-second timer so elapsed/ETA update live.
    this.subs.add(interval(1000).subscribe(() => {
      if (this.sessionStartTs && (this.isRunning() || this.isInitializing() || this.isRecovering())) {
        this.sessionElapsedSec.set(Math.floor((Date.now() - this.sessionStartTs) / 1000));
      }
      const at = this.stream.nextRetryAt();
      if (at) {
        this.reconnectCountdownSec.set(Math.max(0, Math.ceil((at - Date.now()) / 1000)));
      } else {
        this.reconnectCountdownSec.set(0);
      }
    }));
    this.subs.add(interval(2000).subscribe(() => {
      if (this.isRecovering()) this.sessionSvc.getSession(id).subscribe();
    }));

    this.subs.add(this.stream.messageComplete$.subscribe(ev => {
      const d = ev.data as MessageCompleteData;
      this.upsertMessage(d.message);
      this.awaitingCodeResult.set(false);
      if (this.autoScrollEnabled()) this.shouldScrollToBottom = true;
    }));

    this.subs.add(this.stream.agentSwitch$.subscribe(ev => {
      const d = ev.data as AgentSwitchData;
      this.chatItems.update(items => [
        ...items,
        { kind: 'delegation', turn: ev.turn, delegation: { from: d.from_agent, to: d.to_agent, command: d.command } }
      ]);
    }));

    this.subs.add(this.stream.codeSubmitted$.subscribe(ev => {
      const d = ev.data as CodeSubmittedData;
      const key = `${ev.turn}-${d.block_index}`;
      // The server replays the full event history on every WebSocket
      // (re)connect, so a code_submitted event for a block already rendered
      // must not append a second card.
      const alreadyRendered = this.chatItems().some(
        item => item.kind === 'code' && item.turn === ev.turn &&
          item.codeEvent?.submitted.block_index === d.block_index
      );
      this.pendingCode.update(m => { const n = new Map(m); n.set(key, d); return n; });
      if (alreadyRendered) {
        return;
      }
      this.awaitingCodeResult.set(true);
      this.chatItems.update(items => [...items, {
        kind: 'code',
        turn: ev.turn,
        codeEvent: { submitted: d },
      }]);
      if (this.autoScrollEnabled()) this.shouldScrollToBottom = true;
    }));

    this.subs.add(this.stream.codeResult$.subscribe(ev => {
      const result = ev.data as CodeResultData;
      const key = `${ev.turn}-${result.block_index}`;
      const pending = this.pendingCode();
      const submitted = pending.get(key);
      if (submitted) {
        this.pendingCode.update(m => { const n = new Map(m); n.delete(key); return n; });
        this.chatItems.update(items => {
          for (let i = items.length - 1; i >= 0; i--) {
            const item = items[i];
            if (item.kind === 'code' && item.codeEvent && !item.codeEvent.result &&
                item.codeEvent.submitted.block_index === result.block_index && item.turn === ev.turn) {
              const updated = [...items];
              updated[i] = { ...item, codeEvent: { submitted: item.codeEvent.submitted, result } };
              return updated;
            }
          }
          return items;
        });
        if (this.autoScrollEnabled()) this.shouldScrollToBottom = true;
      }
      if (this.pendingCode().size === 0) {
        this.awaitingCodeResult.set(false);
      }
    }));

    this.subs.add(this.stream.artifacts$.subscribe(() => {
      this.sessionSvc.getArtifacts(id).subscribe(a => this.artifacts.set(a));
    }));

    this.subs.add(this.stream.workItemChanges$.subscribe(ev => {
      const d = ev.data as WorkItemChangedData;
      this.workItems.update(items => {
        const summary: WorkItemSummary = d.item;
        return [...items.filter(item => item.id !== summary.id), summary]
          .sort((a, b) => a.id - b.id);
      });
      if (this.selectedWorkItem()?.id === d.item.id) {
        this.selectedWorkItem.set(d.item);
      }
    }));

    this.subs.add(this.stream.errors$.subscribe(ev => {
      this.waitingForAgent.set(false);
      const d = ev.data as ErrorData;
      const record: ErrorRecord = {
        code: d.code,
        message: d.message,
        fatal: d.fatal,
        timestamp: ev.timestamp,
        suggested_fix: d.suggested_fix ?? null,
      };
      this.errorLog.update(errs => [...errs, record]);
      this.chatItems.update(items => [...items, { kind: 'error', turn: ev.turn, error: record }]);
      if (this.autoScrollEnabled()) this.shouldScrollToBottom = true;
      if (!d.fatal && this.prefsSvc.prefs().toastNotifications) {
        this.toasts.show({
          kind: 'warn',
          title: `${d.code}`,
          detail: d.message,
          ttlMs: 6000,
        });
      }
    }));

    this.subs.add(this.stream.statusChanges$.subscribe(ev => {
      const d = ev.data as StatusChangeData;
      if (d.status === 'idle' || d.status === 'stopped' || d.status === 'error') {
        this.waitingForAgent.set(false);
        this.cancellingResponse.set(false);
      }
      if (d.status === 'running' && !this.memoryPollSub) {
        this.startMemoryPolling(id);
      } else if ((d.status === 'stopped' || d.status === 'error') && this.memoryPollSub) {
        this.stopMemoryPolling(id);
      }
      this.statusLog.update(log => {
        const last = log[log.length - 1];
        if (last && last.status === d.status && last.reason === d.reason) {
          return [...log.slice(0, -1), { ...last, count: last.count + 1 }];
        }
        return [...log, { status: d.status, reason: d.reason ?? null, timestamp: ev.timestamp, count: 1 }];
      });
    }));

    this.subs.add(this.stream.systemMessages$.subscribe(ev => {
      const d = ev.data as SystemMessageData;
      this.upsertMessage({
        id: d.id,
        session_id: ev.session_id,
        turn: ev.turn,
        role: 'system',
        agent_name: d.category || 'System',
        content: d.content,
        timestamp: ev.timestamp,
        is_delegation: false,
      });
    }));

    this.subs.add(this.stream.recoveryCompleted$.subscribe(ev => {
      const d = ev.data as RecoveryCompletedData;
      const id = `recovery-${ev.timestamp.replace(/[^a-zA-Z0-9]/g, '-')}`;
      this.chatItems.update(items => {
        if (items.some(item => item.kind === 'recovery' && item.recovery?.id === id)) {
          return items;
        }
        return [
          ...items,
          {
            kind: 'recovery',
            turn: ev.turn,
            recovery: { ...d, id, timestamp: ev.timestamp },
          },
        ];
      });
      if (this.autoScrollEnabled()) this.shouldScrollToBottom = true;
    }));
  }

  ngAfterViewChecked(): void {
    if (this.shouldScrollToBottom && this.autoScrollEnabled()) {
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.stopMemoryPolling();
    this.stream.disconnect();
    document.title = this.originalTitle;
  }

  fetchMemoryState(id: string): void {
    this.sessionSvc.getMemoryState(id).subscribe({
      next: (state) => {
        this.memoryStateError.set(false);
        this.memoryState.set(state);
      },
      error: () => this.memoryStateError.set(true),
    });
  }

  private startMemoryPolling(id: string): void {
    if (!this.memoryActive()) return;
    this.stopMemoryPolling();
    this.memoryPollSub = interval(5000).subscribe(() => this.fetchMemoryState(id));
  }

  private stopMemoryPolling(id?: string): void {
    if (this.memoryPollSub) {
      this.memoryPollSub.unsubscribe();
      this.memoryPollSub = null;
    }
    if (id) this.fetchMemoryState(id);
  }

  sendMessage(): void {
    const content = this.userInput().trim();
    if (!content) return;
    const s = this.session();
    if (!s || s.status !== 'idle' || this.waitingForAgent()) return;

    if (s.status === 'idle' && s.current_turn === 0 && s.mode === 'interactive') {
      this.stream.startRun(content);
    } else {
      this.stream.sendUserMessage(content);
    }
    this.waitingForAgent.set(true);
    this.userInput.set('');
    this.pushHistory(content);
    this.chatItems.update(items => [...items, {
      kind: 'message',
      turn: s.current_turn + 1,
      message: {
        id: 'pending-' + Date.now(),
        session_id: s.id,
        turn: s.current_turn + 1,
        role: 'user',
        agent_name: '',
        content,
        timestamp: new Date().toISOString(),
        is_delegation: false,
      }
    }]);
    this.shouldScrollToBottom = true;
  }

  continueSession(): void {
    const s = this.session();
    if (!s || s.status !== 'idle' || this.waitingForAgent()) return;
    this.stream.sendUserMessage('Please continue with the next step.');
    this.waitingForAgent.set(true);
    this.pushHistory('Continue');
    const content = 'Please continue with the next step.';
    this.chatItems.update(items => [...items, {
      kind: 'message',
      turn: s.current_turn + 1,
      message: {
        id: 'pending-' + Date.now(),
        session_id: s.id,
        turn: s.current_turn + 1,
        role: 'user',
        agent_name: '',
        content,
        timestamp: new Date().toISOString(),
        is_delegation: false,
      }
    }]);
    this.shouldScrollToBottom = true;
  }

  stopSession(): void {
    this.stream.stop();
  }

  endSession(): void {
    if (!window.confirm('End this session? This cannot be undone.')) return;
    this.stream.stop();
  }

  openResume(): void {
    const s = this.session();
    if (!s || s.status !== 'stopped') return;
    this.recoveryForm = {
      name: s.name,
      recovery_mode: 'smart',
      target_mode: s.mode === 'auto' ? 'interactive' : s.mode,
      additional_turns: 20,
      acknowledge_replay_risk: false,
    };
    this.recoveryError.set(null);
    this.recoveryDialog.set('resume');
  }

  openFork(): void {
    const s = this.session();
    if (!s) return;
    this.recoveryForm = {
      name: `${s.name || s.id.slice(0, 8)} fork`,
      llm_backend: s.llm_backend,
      model_name: s.llm_backend === 'openrouter' ? s.resolved_model?.model ?? '' : undefined,
      ollama_model: s.llm_backend === 'ollama' ? s.resolved_model?.model ?? '' : undefined,
      recovery_mode: 'smart',
      target_mode: s.mode === 'auto' ? 'interactive' : s.mode,
      additional_turns: 20,
      acknowledge_replay_risk: false,
    };
    if (s.llm_backend === 'openrouter' && !this.configSvc.openRouterCatalogue()) {
      this.configSvc.getOpenRouterModels().subscribe();
    }
    this.recoveryError.set(null);
    this.recoveryDialog.set('fork');
  }

  closeRecoveryDialog(): void {
    if (!this.recoverySubmitting()) this.recoveryDialog.set(null);
  }

  submitRecovery(): void {
    const s = this.session();
    const kind = this.recoveryDialog();
    if (!s || !kind) return;
    if (kind === 'fork' && !this.recoveryForm.name.trim()) {
      this.recoveryError.set('A name is required for the fork.');
      return;
    }
    if (this.recoveryForm.recovery_mode === 'literal_replay' && !this.recoveryForm.acknowledge_replay_risk) {
      this.recoveryError.set('Acknowledge that literal replay can repeat external side effects.');
      return;
    }
    if (this.recoveryForm.target_mode === 'auto' && !(this.recoveryForm.additional_turns! > 0)) {
      this.recoveryError.set('Enter an additional-turn budget for Auto mode.');
      return;
    }
    const forkTab = kind === 'fork' ? reserveNewTab() : null;
    this.recoverySubmitting.set(true);
    const request: SessionResumeRequest = {
      recovery_mode: this.recoveryForm.recovery_mode,
      target_mode: this.recoveryForm.target_mode,
      additional_turns: this.recoveryForm.target_mode === 'auto' ? this.recoveryForm.additional_turns : undefined,
      acknowledge_replay_risk: this.recoveryForm.acknowledge_replay_risk,
    };
    const operation = kind === 'resume'
      ? this.sessionSvc.resumeSession(s.id, request)
      : kind === 'retry'
        ? this.sessionSvc.retryRecovery(s.id, request)
        : this.sessionSvc.forkSession(s.id, {
            ...request,
            name: this.recoveryForm.name.trim(),
            llm_backend: this.recoveryForm.llm_backend,
            model_name: this.recoveryForm.model_name,
            ollama_model: this.recoveryForm.ollama_model,
            evaluator_model: this.recoveryForm.evaluator_model,
            model_change_reason: this.recoveryForm.model_change_reason?.trim() || undefined,
          });
    operation.subscribe({
      next: result => {
        this.recoverySubmitting.set(false);
        this.recoveryDialog.set(null);
        if (kind === 'fork') {
          if (!navigateTabToSession(forkTab, result.id)) {
            this.toasts.show({
              kind: 'warn',
              title: 'New tab blocked',
              detail: 'The fork was opened in this tab instead.',
              ttlMs: 6000,
            });
            void this.router.navigate(['/session', result.id]);
          }
        } else {
          this.stream.connect(s.id);
        }
      },
      error: err => {
        forkTab?.close();
        this.recoverySubmitting.set(false);
        this.recoveryError.set(err?.error?.detail ?? `Unable to ${kind} this session.`);
      },
    });
  }

  retryRecovery(mode: RecoveryMode): void {
    const s = this.session();
    if (!s) return;
    this.recoveryForm.recovery_mode = mode;
    this.recoveryForm.acknowledge_replay_risk = mode !== 'literal_replay';
    this.recoveryError.set(null);
    this.recoveryDialog.set('retry');
  }

  onRecoveryBackendChange(backend: string): void {
    this.recoveryForm.llm_backend = backend;
    this.recoveryForm.model_name = undefined;
    this.recoveryForm.ollama_model = undefined;
    if (backend === 'openrouter') {
      this.configSvc.getOpenRouterModels().subscribe(catalogue => {
        this.recoveryForm.model_name = catalogue.models[0]?.canonical_slug;
      });
    } else if (backend === 'ollama') {
      this.configSvc.getOllamaModels().subscribe(result => {
        this.recoveryForm.ollama_model = result.default_model || result.models[0];
      });
    }
  }

  acceptPartialRecovery(): void {
    const s = this.session();
    if (!s) return;
    this.sessionSvc.acceptPartialRecovery(s.id).subscribe({
      next: () => this.stream.connect(s.id),
      error: err => this.recoveryError.set(err?.error?.detail ?? 'Unable to continue partial recovery.'),
    });
  }

  cancelResponse(): void {
    if (!this.isRunning() || this.session()?.mode !== 'interactive') return;
    this.cancellingResponse.set(true);
    this.stream.cancelResponse();
  }

  evaluate(): void {
    const id = this.session()?.id;
    if (!id || !this.canEvaluate() || this.evaluating()) return;
    this.evaluating.set(true);
    this.evaluationError.set(null);
    this.sessionSvc.evaluate(id).subscribe({
      next: (result) => {
        this.evaluating.set(false);
        this.evaluationResult.set(result);
      },
      error: (err) => {
        this.evaluating.set(false);
        this.evaluationError.set(err?.error?.detail ?? 'Evaluation failed.');
        this.toasts.show({ kind: 'error', title: 'Evaluation failed', ttlMs: 4000 });
      },
    });
  }

  loadWorkItems(sessionId?: string): void {
    const id = sessionId ?? this.session()?.id;
    if (!id) return;
    this.sessionSvc.getWorkItems(id).subscribe({
      next: items => this.workItems.set(items),
      error: err => this.workItemError.set(err?.error?.detail ?? 'Unable to load work items.'),
    });
  }

  selectWorkItem(itemId: number): void {
    const id = this.session()?.id;
    if (!id) return;
    this.workItemError.set(null);
    this.sessionSvc.getWorkItem(id, itemId).subscribe({
      next: item => this.selectedWorkItem.set(item),
      error: err => this.workItemError.set(err?.error?.detail ?? 'Unable to load work item.'),
    });
  }

  canReviewWorkItem(item: WorkItemSummary): boolean {
    return this.canEvaluate() && (item.status === 'In review' || item.status === 'Done');
  }

  reviewWorkItem(itemId: number): void {
    const id = this.session()?.id;
    if (!id || this.workItemReviewing()) return;
    this.workItemReviewing.set(true);
    this.workItemError.set(null);
    this.sessionSvc.reviewWorkItem(id, itemId).subscribe({
      next: result => {
        this.workItemReviewing.set(false);
        this.selectedWorkItem.set(result.item);
        this.workItems.update(items => [
          ...items.filter(item => item.id !== result.item.id),
          result.item,
        ].sort((a, b) => a.id - b.id));
      },
      error: err => {
        this.workItemReviewing.set(false);
        this.workItemError.set(err?.error?.detail ?? 'Work-item review failed.');
      },
    });
  }

  openEvaluatorModelEditor(): void {
    const state = this.session()?.evaluator_model;
    if (!state) return;
    this.evaluatorModelForm = { ...state.selection };
    this.evaluatorModelReason.set('');
    this.evaluatorModelError.set(null);
    this.editingEvaluatorModel.set(true);
  }

  saveEvaluatorModel(): void {
    const session = this.session();
    if (!session || this.evaluatorModelSaving()) return;
    if (
      this.evaluatorModelForm.mode === 'explicit' &&
      (!this.evaluatorModelForm.llm_backend?.trim() || !this.evaluatorModelForm.model_name?.trim())
    ) {
      this.evaluatorModelError.set('Backend and exact model identifier are required.');
      return;
    }
    const selection: EvaluatorModelConfig = this.evaluatorModelForm.mode === 'inherit_worker'
      ? { mode: 'inherit_worker' }
      : {
          mode: 'explicit',
          llm_backend: this.evaluatorModelForm.llm_backend?.trim(),
          model_name: this.evaluatorModelForm.model_name?.trim(),
        };
    this.evaluatorModelSaving.set(true);
    this.evaluatorModelError.set(null);
    this.sessionSvc.updateEvaluatorModel(session.id, {
      selection,
      expected_revision: session.evaluator_model.revision,
      reason: this.evaluatorModelReason().trim() || null,
    }).subscribe({
      next: () => {
        this.evaluatorModelSaving.set(false);
        this.editingEvaluatorModel.set(false);
        this.toasts.show({ kind: 'success', title: 'Evaluator model updated', ttlMs: 3000 });
      },
      error: err => {
        this.evaluatorModelSaving.set(false);
        this.evaluatorModelError.set(err?.error?.detail ?? 'Unable to update evaluator model.');
      },
    });
  }

  retryReconnect(): void {
    this.stream.retryNow();
  }

  copyToClipboard(text: string): void {
    try {
      navigator.clipboard.writeText(text);
      this.toasts.show({ kind: 'success', title: 'Copied to clipboard', ttlMs: 2000 });
    } catch {}
  }

  copyErrorToClipboard(err: ErrorRecord): void {
    const payload = [
      `[${err.code}] ${err.message}`,
      err.suggested_fix ? `Suggested fix: ${err.suggested_fix}` : null,
      `At: ${err.timestamp}`,
    ].filter(Boolean).join('\n');
    this.copyToClipboard(payload);
  }

  formatDuration(seconds: number | null): string {
    if (seconds === null) return '—';
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    if (m < 60) return `${m}m ${s.toString().padStart(2, '0')}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${(m % 60).toString().padStart(2, '0')}m`;
  }

  formatMs(ms: number): string {
    if (!isFinite(ms) || ms <= 0) return '—';
    return this.formatDuration(Math.round(ms / 1000));
  }

  isSlowBlock(codeEvent: { result?: CodeResultData }): boolean {
    const ms = codeEvent.result?.duration_ms ?? 0;
    return ms >= this.prefsSvc.prefs().slowCodeThresholdMs;
  }

  jumpToAnchor(selector: string): void {
    this.olderConversationExpanded.set(true);
    setTimeout(() => {
      const el = document.querySelector(selector);
      if (!el) return;
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      this.autoScrollEnabled.set(false);
      this.showJumpToLatest.set(true);
    });
  }

  recoveryModeLabel(mode: string | null | undefined): string {
    if (mode === 'smart') return 'Smart rebuild';
    if (mode === 'literal_replay') return 'Literal replay';
    return 'Best effort';
  }

  jumpToLatest(): void {
    this.autoScrollEnabled.set(true);
    this.showJumpToLatest.set(false);
    this.scrollToBottom();
  }

  toggleOlderConversation(): void {
    this.olderConversationExpanded.update(v => !v);
  }

  toggleTimeline(): void {
    this.showTimeline.update(v => !v);
  }

  toggleContext(): void {
    this.showContext.update(v => {
      if (!v) this.fetchMemoryState(this.route.snapshot.paramMap.get('id')!);
      return !v;
    });
  }

  retryMemoryState(): void {
    this.fetchMemoryState(this.route.snapshot.paramMap.get('id')!);
  }

  goBack(): void {
    this.router.navigate(['/']);
  }

  private scrollToBottom(): void {
    try {
      this.chatBottom.nativeElement.scrollIntoView({ behavior: 'smooth' });
    } catch { /* noop */ }
  }

  onScroll(event: Event): void {
    const el = event.target as HTMLElement;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const wasEnabled = this.autoScrollEnabled();
    const nearBottom = distanceFromBottom < AUTO_SCROLL_THRESHOLD_PX;
    // Only flip state on change to avoid signal churn.
    if (nearBottom && !wasEnabled) {
      this.autoScrollEnabled.set(true);
      this.showJumpToLatest.set(false);
    } else if (!nearBottom && wasEnabled) {
      this.autoScrollEnabled.set(false);
      this.showJumpToLatest.set(true);
    }
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
      return;
    }
    // Input history — only when caret in empty single-line context
    if (event.key === 'ArrowUp' && !event.shiftKey && !this.userInput().includes('\n') && this.inputHistory.length) {
      if (this.historyIndex === -1) this.historyDraft = this.userInput();
      this.historyIndex = Math.min(this.historyIndex + 1, this.inputHistory.length - 1);
      this.userInput.set(this.inputHistory[this.inputHistory.length - 1 - this.historyIndex] ?? '');
      event.preventDefault();
      return;
    }
    if (event.key === 'ArrowDown' && !event.shiftKey && this.historyIndex >= 0) {
      this.historyIndex--;
      if (this.historyIndex === -1) this.userInput.set(this.historyDraft);
      else this.userInput.set(this.inputHistory[this.inputHistory.length - 1 - this.historyIndex] ?? '');
      event.preventDefault();
    }
  }

  @HostListener('window:keydown', ['$event'])
  onGlobalKeyDown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    const editable = target && (
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable
    );
    // Esc — stop run
    if (event.key === 'Escape' && this.isRunning()) {
      event.preventDefault();
      this.stopSession();
      this.toasts.show({ kind: 'info', title: 'Stopping session…', ttlMs: 2000 });
      return;
    }
    // Cmd/Ctrl-K — focus input
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      this.messageInput?.nativeElement?.focus();
      return;
    }
    // Cmd/Ctrl-/ — show shortcuts
    if ((event.metaKey || event.ctrlKey) && event.key === '/') {
      event.preventDefault();
      this.showShortcutHelp.update(v => !v);
      return;
    }
    // "/" outside editable — focus input
    if (event.key === '/' && !editable) {
      event.preventDefault();
      this.messageInput?.nativeElement?.focus();
    }
  }

  private notifyCompletion(status: string): void {
    const prefs = this.prefsSvc.prefs();
    if (!prefs.toastNotifications) return;
    const stopped = status === 'stopped';
    this.toasts.show({
      kind: stopped ? 'success' : 'error',
      title: stopped ? 'Session complete' : 'Session ended with an error',
      detail: stopped ? 'The agent finished its run.' : this.lastError()?.message ?? undefined,
      ttlMs: 8000,
    });
    if (prefs.soundOnComplete) this.playBeep(stopped);
  }

  private playBeep(good: boolean): void {
    try {
      const AC = (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!AC) return;
      const ctx = new AC();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = good ? 880 : 260;
      gain.gain.value = 0.05;
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      setTimeout(() => { osc.stop(); ctx.close(); }, 180);
    } catch {}
  }

  private upsertMessage(message: Message): void {
    this.chatItems.update(items => {
      if (items.some(item => item.kind === 'message' && item.message?.id === message.id)) {
        return items;
      }
      const pendingIndex = items.findIndex(item =>
        item.kind === 'message' &&
        item.message?.id.startsWith('pending-') &&
        item.message.role === message.role &&
        item.message.content === message.content
      );
      if (pendingIndex >= 0) {
        return items.map((item, index) =>
          index === pendingIndex ? { kind: 'message', message, turn: message.turn } : item
        );
      }
      return [...items, { kind: 'message', message, turn: message.turn }];
    });
  }

  private loadHistory(): string[] {
    try {
      const raw = localStorage.getItem(INPUT_HISTORY_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  private pushHistory(entry: string): void {
    const trimmed = entry.trim();
    if (!trimmed) return;
    // Dedup consecutive identical entries.
    if (this.inputHistory[this.inputHistory.length - 1] === trimmed) return;
    this.inputHistory.push(trimmed);
    if (this.inputHistory.length > INPUT_HISTORY_LIMIT) {
      this.inputHistory.splice(0, this.inputHistory.length - INPUT_HISTORY_LIMIT);
    }
    this.historyIndex = -1;
    this.historyDraft = '';
    try { localStorage.setItem(INPUT_HISTORY_KEY, JSON.stringify(this.inputHistory)); } catch {}
  }
}
