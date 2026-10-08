import {
  Component, OnInit, OnDestroy, inject, signal, ViewChild,
  ElementRef, AfterViewChecked, computed, HostListener, effect, afterNextRender, Injector, untracked,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { Subscription, interval, map } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';
import { SessionService } from '../../core/services/session.service';
import { ConfigService } from '../../core/services/config.service';
import { AgentStreamService } from '../../core/services/agent-stream.service';
import { ToastService } from '../../core/services/toast.service';
import { PreferencesService } from '../../core/services/preferences.service';
import { SessionStore } from '../../core/state/session-store.service';
import { ChatItem, ErrorRecord } from '../../core/state/session-state.model';
import { artifactsByAction } from '../../core/utils/artifacts';
import {
  Artifact, MemoryState, EvaluationResult, EvaluatorModelConfig,
  RecoveryMode, SessionForkRequest, SessionResumeRequest,
  WorkItemSummary, SessionBriefFields,
} from '../../core/models/session.model';
import { MessageBubbleComponent } from '../../shared/components/message-bubble/message-bubble';
import { CodeCardComponent } from '../../shared/components/code-card/code-card';
import { ArtifactCardComponent } from '../../shared/components/artifact-card/artifact-card';
import { StatusIndicatorComponent } from '../../shared/components/status-indicator/status-indicator';
import { IconComponent } from '../../shared/components/icon/icon';
import { TooltipDirective } from '../../shared/directives/tooltip.directive';
import { navigateTabToSession, reserveNewTab } from '../../core/utils/app-navigation';
import { BlockMessage, WorkbenchComponent } from './workbench/workbench';
import { SessionView, ViewToggleComponent } from './workbench/view-toggle';
import { anchorLabel } from './workbench/block-actions';
import { QueueBarComponent } from './queue-bar/queue-bar';
import { SplitPaneComponent } from './split-pane/split-pane';
import { SPLIT_MIN_VIEWPORT_PX } from './split-pane/split-ratio';
import { blockIdFromFragment } from './workbench/block-matching';
import { BranchBannerComponent } from './branch-banner/branch-banner';

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
    MessageBubbleComponent, CodeCardComponent, ArtifactCardComponent, StatusIndicatorComponent,
    IconComponent, TooltipDirective, WorkbenchComponent, ViewToggleComponent, BranchBannerComponent,
    QueueBarComponent, SplitPaneComponent,
  ],
  // One store per session page: a fresh page starts from empty state.
  providers: [SessionStore],
  templateUrl: './session.html',
  styleUrl: './session.scss',
})
export class SessionComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('chatBottom') chatBottom!: ElementRef;
  @ViewChild('chatPanel') chatPanel!: ElementRef<HTMLElement>;
  @ViewChild('messageInput') messageInput!: ElementRef<HTMLTextAreaElement>;
  readonly COMPACT_AFTER_ITEMS = COMPACT_AFTER_ITEMS;
  readonly anchorLabel = anchorLabel;

  private route = inject(ActivatedRoute);
  private router = inject(Router);
  sessionSvc = inject(SessionService);
  configSvc = inject(ConfigService);
  stream = inject(AgentStreamService);
  private toasts = inject(ToastService);
  private prefsSvc = inject(PreferencesService);
  private store = inject(SessionStore);

  // Event-derived state lives in the store; these aliases keep the template's
  // names.
  chatItems = this.store.chatItems;
  artifacts = this.store.artifacts;
  errorLog = this.store.errorLog;
  statusLog = this.store.statusLog;
  waitingForAgent = this.store.waitingForAgent;
  cancellingResponse = this.store.cancellingResponse;
  awaitingCodeResult = this.store.awaitingCodeResult;
  workItems = this.store.workItems;
  selectedWorkItem = this.store.selectedWorkItem;
  briefDraft = this.store.briefDraft;
  briefEditDraft = this.store.briefEditDraft;
  briefEditing = this.store.briefEditing;
  submittingBriefDecision = this.store.submittingBriefDecision;
  lastError = this.store.lastError;
  hasRecoveryMilestone = this.store.hasRecoveryMilestone;

  olderConversationExpanded = signal(false);
  userInput = signal('');
  showTimeline = signal(false);
  artifactFilter = signal<ArtifactFilter>('all');
  artifactSearch = signal('');
  memoryState = signal<MemoryState | null>(null);
  memoryStateError = signal(false);
  evaluating = signal(false);
  evaluationResult = signal<EvaluationResult | null>(null);
  evaluationError = signal<string | null>(null);
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

  session = this.sessionSvc.currentSession;
  // `?view=workbench` shows the read-only workbench; no param is the chat.
  // Both views render from the one store and the one WebSocket.
  private viewParam = toSignal(this.route.queryParamMap.pipe(map(p => p.get('view'))),
    { initialValue: this.route.snapshot.queryParamMap.get('view') });
  // The split view needs a wide window; below it `?view=split` shows the chat
  // (with a toast saying why, see the constructor).
  private splitQuery = window.matchMedia(`(min-width: ${SPLIT_MIN_VIEWPORT_PX}px)`);
  splitAvailable = signal(this.splitQuery.matches);
  view = computed<SessionView>(() => {
    const param = this.viewParam();
    if (param === 'workbench') return 'workbench';
    if (param === 'split') return this.splitAvailable() ? 'split' : 'chat';
    return 'chat';
  });
  /** The sidebar as an overlay drawer in the split view. */
  splitDetailsOpen = signal(false);
  private injector = inject(Injector);
  private fragment = toSignal(this.route.fragment, { initialValue: this.route.snapshot.fragment });
  /** The workbench's selected block (`#block:<id>`). */
  private selectedBlockId = computed(() => blockIdFromFragment(this.fragment()));
  /** In the split view, the selected block's action ids (its code cards are highlighted). */
  private linkedActionIds = computed(() => {
    const id = this.selectedBlockId();
    if (this.view() !== 'split' || id === null) return null;
    const block = this.store.blocks().find(b => b.block_id === id);
    return block ? new Set(block.action_ids) : null;
  });
  // A selection made by clicking a code card in the chat: don't scroll the chat.
  private selectionFromChat: string | null = null;
  private lastScrolledBlockId: string | null = null;
  // WS-5: the briefing conversation is a distinct phase, not a normal chat
  // turn — `briefPhase` gates the interview view (the draft is in the store).
  briefPhase = computed(() => this.session()?.phase === 'briefing');
  frozenBrief = computed(() => this.session()?.brief ?? null);
  status = computed(() => this.session()?.status ?? 'stopped');
  currentAgent = computed(() => this.session()?.current_agent ?? '');
  isIdle = computed(() => this.status() === 'idle');
  isRunning = computed(() => this.status() === 'running');
  isStopped = computed(() => this.status() === 'stopped');
  isError = computed(() => this.status() === 'error');
  isInitializing = computed(() => this.status() === 'initializing');
  isRecovering = computed(() => this.status() === 'recovering');
  messageQueue = this.store.messageQueue;
  pendingRemovals = this.store.pendingRemovals;
  // Whether the session takes a user message now: the chat input and the
  // workbench's block composer both use this.
  // The server sends a message straight through only when its queue is
  // empty; with messages waiting, a new one joins the end of the queue.
  canSendNow = computed(() =>
    this.status() === 'idle' && !this.waitingForAgent() && this.messageQueue().length === 0);
  // Whether a message sent now is queued on the server instead: an
  // interactive session past its first turn whose agent is busy. The first
  // turn's `run` message is never queued.
  canQueue = computed(() => {
    const s = this.session();
    const status = this.status();
    return !!s && s.mode === 'interactive' && s.current_turn > 0 &&
      (status === 'running' || status === 'initializing' || status === 'recovering' ||
        (status === 'idle' && this.messageQueue().length > 0));
  });
  /** The input takes text when a message can be sent now or queued. */
  canWriteMessage = computed(() => this.canSendNow() || this.canQueue());

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
  /** Artifacts by the code action that last wrote them, for the code cards. */
  artifactsByAction = computed(() => artifactsByAction(this.artifacts()));
  private readonly NO_ARTIFACTS: Artifact[] = [];
  /** The artifacts a code card's action produced; none for a card without an action id. */
  codeArtifacts(actionId: string | undefined): Artifact[] {
    return (actionId ? this.artifactsByAction().get(actionId) : undefined) ?? this.NO_ARTIFACTS;
  }
  /**
   * `@for` track key for the chat. A code item keeps its key when its result
   * replaces the item object, and every key survives the visible window
   * sliding (new items, "show earlier"), so a code card keeps its expanded
   * state. Code items key on their action id (or the turn and block index the
   * result is matched on); other items on their position in the full list.
   */
  chatItemKey(item: ChatItem, visibleIndex: number): string {
    const submitted = item.kind === 'code' ? item.codeEvent?.submitted : undefined;
    if (submitted) {
      return submitted.action_id
        ? `code:${submitted.action_id}`
        : `code:${item.turn}:${submitted.block_index}`;
    }
    return `item:${this.hiddenChatItemCount() + visibleIndex}`;
  }
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
    effect(() => {
      const param = this.viewParam();
      if (param !== null && param !== 'workbench' && param !== 'chat' && param !== 'split') {
        this.toasts.show({ kind: 'error', title: `Unknown view "${param}"`, detail: 'Showing the chat view.', ttlMs: 6000 });
      }
    });
    // Track the window width for the split view, and say when it falls back.
    const onSplitQuery = (e: MediaQueryListEvent) => this.splitAvailable.set(e.matches);
    this.splitQuery.addEventListener('change', onSplitQuery);
    this.subs.add(() => this.splitQuery.removeEventListener('change', onSplitQuery));
    effect(() => {
      if (this.viewParam() === 'split' && !this.splitAvailable()) {
        untracked(() => this.toasts.show({
          kind: 'warn',
          title: 'Split view unavailable',
          detail: `Split view needs a window at least ${SPLIT_MIN_VIEWPORT_PX}px wide. Showing the chat view.`,
          ttlMs: 8000,
        }));
      }
    });
    // Split view: selecting a block highlights its code cards in the chat and
    // scrolls to the first one.
    effect(() => {
      // Only a change of selection scrolls: block_changed events while the
      // block is selected must not pull the chat back to it.
      const blockId = this.view() === 'split' ? this.selectedBlockId() : null;
      if (blockId === null) {
        this.lastScrolledBlockId = null;
        return;
      }
      if (blockId === this.lastScrolledBlockId) return;
      this.lastScrolledBlockId = blockId;
      if (this.selectionFromChat === blockId) {
        this.selectionFromChat = null;
        return;
      }
      untracked(() => {
        const ids = this.linkedActionIds();
        if (ids === null || ids.size === 0) return;
        const items = this.store.chatItems();
        const first = items.findIndex(item =>
          item.kind === 'code' && ids.has(item.codeEvent?.submitted.action_id ?? ''));
        if (first < 0) return;
        // The card may be in the collapsed older conversation: expand it first.
        if (first < this.hiddenChatItemCount()) this.olderConversationExpanded.set(true);
        const actionId = items[first].codeEvent!.submitted.action_id!;
        afterNextRender(() => {
          const card = this.chatPanel?.nativeElement.querySelector(`[data-action-id="${CSS.escape(actionId)}"]`);
          if (!card) throw new Error(`Code card for action ${actionId} is not rendered.`);
          card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, { injector: this.injector });
      });
    });
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
  }

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id')!;
    // Hydrate from sessionStorage FIRST so refresh isn't jarring.
    this.store.attach(id);

    this.sessionSvc.getSession(id).subscribe({
      next: session => {
        // Before connecting, so replayed message_queue_changed events win.
        this.store.hydrateMessageQueue(session.message_queue);
        // Validate the REST resource before opening a socket. Otherwise a stale
        // deep link briefly enters the WebSocket "expired" state before routing
        // back to the dashboard.
        this.stream.connect(id);
        this.sessionSvc.getMessages(id).subscribe(msgs => {
          // Only replace chat when server has more/newer messages than cache.
          if (this.store.mergeServerMessages(msgs)) {
            this.shouldScrollToBottom = true;
          }
        });
        this.store.refreshArtifacts();
        this.loadWorkItems(id);
        this.store.hydrateBlocks();
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

    // Every session event goes through the store's reducer; what is left
    // here are the side effects that need this component.
    this.subs.add(this.stream.events$.subscribe(ev => {
      const outcome = this.store.apply(ev);
      if (outcome.status !== null) this.followStatusForMemoryPolling(outcome.status, id);
      if (outcome.scroll && this.autoScrollEnabled()) this.shouldScrollToBottom = true;
      if (outcome.nonFatalError && this.prefsSvc.prefs().toastNotifications) {
        this.toasts.show({
          kind: 'warn',
          title: `${outcome.nonFatalError.code}`,
          detail: outcome.nonFatalError.message,
          ttlMs: 6000,
        });
      }
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

  private followStatusForMemoryPolling(status: string, id: string): void {
    if (status === 'running' && !this.memoryPollSub) {
      this.startMemoryPolling(id);
    } else if ((status === 'stopped' || status === 'error') && this.memoryPollSub) {
      this.stopMemoryPolling(id);
    }
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
    if (!this.canWriteMessage()) return;
    if (!this.submitUserMessage(content)) return;
    this.userInput.set('');
  }

  /**
   * A block-scoped message from the workbench. It goes through the same path
   * as a typed message, so it shows in the chat. The workbench only emits
   * while it can send now or queue, so a refusal here is a bug: raise it.
   */
  sendBlockMessage(message: BlockMessage): void {
    if (!this.canWriteMessage()) {
      throw new Error('The session cannot take a message right now.');
    }
    this.submitUserMessage(message.content, message.blockId);
  }

  /**
   * The one send path for user messages (chat input and workbench). A
   * workbench message carries its block's id, which focuses the agent's next
   * code on that block; chat messages carry none. Returns false when a
   * message to be queued was not sent (no connection; a toast says so).
   */
  private submitUserMessage(content: string, blockId?: string): boolean {
    const s = this.session();
    if (!s) throw new Error('No session to send a message to.');
    if (!this.canSendNow()) {
      if (!this.canQueue()) {
        throw new Error('The session cannot take or queue a message right now.');
      }
      // The agent is busy: the server queues this user_message and delivers it
      // when the agent is ready. Its message_queue_changed event shows it in
      // the queue bar; it reaches the chat only once delivered. Nothing else
      // would show a send dropped on a closed socket, so refuse it visibly.
      if (this.stream.connectionState() !== 'open') {
        this.toasts.show({
          kind: 'error',
          title: 'Message not queued',
          detail: 'Not connected to the server. Your text is kept; try again once the connection is back.',
          ttlMs: 8000,
        });
        return false;
      }
      this.stream.sendUserMessage(content, blockId);
      this.pushHistory(content);
      return true;
    }
    if (s.current_turn === 0 && s.mode === 'interactive') {
      // A 'run' message has no block focus; no block exists before the first turn.
      if (blockId !== undefined) {
        throw new Error(`Block ${blockId} cannot be focused before the session has started.`);
      }
      this.stream.startRun(content);
    } else {
      this.stream.sendUserMessage(content, blockId);
    }
    this.pushHistory(content);
    this.store.sendUserMessage(s.id, s.current_turn + 1, content, blockId ?? null);
    this.shouldScrollToBottom = true;
    return true;
  }

  continueSession(): void {
    const s = this.session();
    if (!s || !this.canSendNow()) return;
    const content = 'Please continue with the next step.';
    this.stream.sendUserMessage(content);
    this.pushHistory('Continue');
    this.store.sendUserMessage(s.id, s.current_turn + 1, content);
    this.shouldScrollToBottom = true;
  }

  /** Split view: whether a chat code card belongs to the selected block. */
  isLinkedCode(item: ChatItem): boolean {
    const actionId = item.codeEvent?.submitted.action_id;
    return !!actionId && (this.linkedActionIds()?.has(actionId) ?? false);
  }

  /**
   * Split view: a click on a chat code card (or a plot in it) selects its
   * block in the workbench pane, matched by the card's block_id, else by
   * the block whose action_ids hold its action_id.
   */
  selectBlockForCode(item: ChatItem): void {
    if (this.view() !== 'split') return;
    const submitted = item.codeEvent?.submitted;
    if (!submitted) return;
    const blocks = this.store.blocks();
    const block = (submitted.block_id ? blocks.find(b => b.block_id === submitted.block_id) : undefined)
      ?? (submitted.action_id ? blocks.find(b => b.action_ids.includes(submitted.action_id!)) : undefined);
    if (!block) {
      this.toasts.show({
        kind: 'info',
        title: 'No step for this code',
        detail: 'This code is not part of a recorded workbench step.',
        ttlMs: 4000,
      });
      return;
    }
    if (this.selectedBlockId() === block.block_id) return;
    this.selectionFromChat = block.block_id;
    this.router.navigate([], {
      relativeTo: this.route,
      queryParamsHandling: 'preserve',
      fragment: `block:${block.block_id}`,
      replaceUrl: true,
    });
  }

  /** The queue bar's ✕: ask the server to drop a queued message. */
  unqueueMessage(id: string): void {
    if (this.pendingRemovals().has(id)) return;
    if (this.stream.connectionState() !== 'open') {
      this.toasts.show({
        kind: 'error',
        title: 'Could not remove the queued message',
        detail: 'Not connected to the server. Try again once the connection is back.',
        ttlMs: 6000,
      });
      return;
    }
    this.store.markRemovalPending(id);
    this.stream.send({ type: 'unqueue_message', id });
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
    this.store.markCancelling();
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
      next: items => this.store.setWorkItems(items),
      error: err => this.workItemError.set(err?.error?.detail ?? 'Unable to load work items.'),
    });
  }

  selectWorkItem(itemId: number): void {
    const id = this.session()?.id;
    if (!id) return;
    this.workItemError.set(null);
    this.sessionSvc.getWorkItem(id, itemId).subscribe({
      next: item => this.store.setSelectedWorkItem(item),
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
        this.store.setSelectedWorkItem(result.item);
        this.store.upsertWorkItem(result.item);
      },
      error: err => {
        this.workItemReviewing.set(false);
        this.workItemError.set(err?.error?.detail ?? 'Work-item review failed.');
      },
    });
  }

  acceptBrief(): void {
    const id = this.session()?.id;
    if (!id || this.submittingBriefDecision()) return;
    this.store.setSubmittingBriefDecision(true);
    this.sessionSvc.submitBriefDecision(id, 'accept').subscribe({
      error: err => {
        this.store.setSubmittingBriefDecision(false);
        this.toasts.show({
          kind: 'error',
          title: 'Failed to accept the brief',
          detail: err?.error?.detail,
          ttlMs: 4000,
        });
      },
    });
  }

  rejectBrief(reason: string): void {
    const id = this.session()?.id;
    if (!id || this.submittingBriefDecision()) return;
    this.store.setSubmittingBriefDecision(true);
    this.sessionSvc.submitBriefDecision(id, 'reject', { reason }).subscribe({
      next: () => {
        this.store.setSubmittingBriefDecision(false);
        this.store.clearBriefDraft();
      },
      error: err => {
        this.store.setSubmittingBriefDecision(false);
        this.toasts.show({
          kind: 'error',
          title: 'Failed to reject the brief',
          detail: err?.error?.detail,
          ttlMs: 4000,
        });
      },
    });
  }

  submitEditedBrief(): void {
    const id = this.session()?.id;
    if (!id || this.submittingBriefDecision()) return;
    let parsed: Partial<SessionBriefFields>;
    try {
      parsed = JSON.parse(this.briefEditDraft());
    } catch {
      this.toasts.show({ kind: 'error', title: 'Edited brief is not valid JSON', ttlMs: 4000 });
      return;
    }
    this.store.setSubmittingBriefDecision(true);
    this.sessionSvc.submitBriefDecision(id, 'edit', { brief: parsed }).subscribe({
      error: err => {
        this.store.setSubmittingBriefDecision(false);
        this.toasts.show({
          kind: 'error',
          title: 'Failed to submit the edited brief',
          detail: err?.error?.detail,
          ttlMs: 4000,
        });
      },
    });
  }

  setBriefEditDraft(text: string): void {
    this.store.setBriefEditDraft(text);
  }

  setBriefEditing(editing: boolean): void {
    this.store.setBriefEditing(editing);
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

  setView(view: SessionView): void {
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { view: view === 'chat' ? null : view },
      queryParamsHandling: 'merge',
      preserveFragment: true,
    });
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
