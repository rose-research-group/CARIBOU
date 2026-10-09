import { Component, ElementRef, Injector, afterNextRender, computed, effect, inject, input, output, signal, untracked } from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router } from '@angular/router';
import { forkJoin } from 'rxjs';
import { SessionStore } from '../../../core/state/session-store.service';
import { Block } from '../../../core/models/block.model';
import {
  Artifact, ReviewEvidence, ReviewEvidenceBlock, Session, WorkItemAnchor, WorkItemDetail, WorkItemReview,
  WorkItemSummary,
} from '../../../core/models/session.model';
import { BlueprintContent } from '../../../core/models/blueprint.model';
import { SessionService } from '../../../core/services/session.service';
import { AgentStreamService } from '../../../core/services/agent-stream.service';
import { ConfigService } from '../../../core/services/config.service';
import { ToastService } from '../../../core/services/toast.service';
import { artifactPreviewUrl, artifactsByAction } from '../../../core/utils/artifacts';
import { CodeCardComponent } from '../../../shared/components/code-card/code-card';
import { ArtifactCardComponent } from '../../../shared/components/artifact-card/artifact-card';
import { IconComponent } from '../../../shared/components/icon/icon';
import { PlotCarouselComponent } from '../../../shared/components/plot-carousel/plot-carousel';
import {
  STATUS_GLYPH, STATUS_LABEL, blockActions, blockArtifacts, blockIdFromFragment,
  blockPlots, plural, turnSpan, workItemLabel,
} from './block-matching';
import {
  BlockReference, anchorLabel, canHumanReview, formatBlockMessage, httpErrorMessage,
  referenceKey, sentBackText,
} from './block-actions';
import { RESTORE_MODE_LABEL, shortId } from './branching';
import { GATE_GLYPH, GATE_LABEL, Gate, blockGate, gateIdFromFragment } from './gates';
import { activityLine } from './activity';
import { BranchFlow, FlowCell, branchFlow, lastColumn, ownCellsWithoutParent, sessionCells } from './flow-layout';
import { rerunLabel, rerunLinks, shownVersion, versionText } from './stages';
import { GhostCardComponent } from './ghost-card/ghost-card';
import { GateNodeComponent } from './gate-node/gate-node';
import { BranchFormComponent } from './branch-form/branch-form';
import { PathsRowComponent } from './paths-row/paths-row';
import { BranchPaths } from './paths-row/branch-paths';

type Section = 'workitem' | 'code' | 'artifacts';
type ReviewAction = 'approve' | 'reject' | 'evaluator';

/** An open "Open ticket" form: where it was opened (`origin`) and its anchor. */
interface TicketForm {
  origin: string;
  anchor: WorkItemAnchor;
}

/**
 * Workbench view of a session: the Goal, one card per block in `index` order,
 * a foldable Plots row, and a side panel for the selected block. Everything
 * comes from the page's SessionStore; the selection is the URL fragment
 * (`#block:<block_id>`), so it deep-links.
 *
 * Tier 1 (steer and annotate): the panel can message the agent about the
 * block, approve or send back its work item, run the evaluator review and
 * open tickets. Each action is a user message (sent through the page's chat
 * path) or a REST work-item change; the resulting `work_item_changed` events
 * update the store, so nothing here patches state itself.
 */
/** A message about one block: the text the agent reads and the block it focuses. */
export interface BlockMessage {
  content: string;
  blockId: string;
}

@Component({
  selector: 'app-workbench',
  standalone: true,
  imports: [
    CodeCardComponent, ArtifactCardComponent, IconComponent, NgTemplateOutlet, PlotCarouselComponent,
    BranchFormComponent, PathsRowComponent, GhostCardComponent, GateNodeComponent,
  ],
  // The branch lanes; the grid widens to the longest one.
  providers: [BranchPaths],
  templateUrl: './workbench.html',
  styleUrl: './workbench.scss',
})
export class WorkbenchComponent {
  private store = inject(SessionStore);
  private stream = inject(AgentStreamService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private sessionSvc = inject(SessionService);
  private configSvc = inject(ConfigService);
  private toasts = inject(ToastService);
  private injector = inject(Injector);
  readonly paths = inject(BranchPaths);

  /** A code card's Copy button; the page owns the clipboard and its toast. */
  readonly copySource = output<string>();
  /** Whether the session takes a user message now (the chat input's condition). */
  readonly canSend = input.required<boolean>();
  /** Whether a message sent now is queued until the agent is ready (the agent is busy). */
  readonly canQueue = input.required<boolean>();
  /** The composer takes text when a message can be sent now or queued. */
  readonly canWrite = computed(() => this.canSend() || this.canQueue());
  /** A block-scoped message; the page sends it through the chat's send path. */
  readonly sendMessage = output<BlockMessage>();
  /** Split view: scroll the selected block into view when the selection changes. */
  readonly scrollSelectionIntoView = input(false);
  private host = inject(ElementRef<HTMLElement>);

  readonly STATUS_GLYPH = STATUS_GLYPH;
  readonly STATUS_LABEL = STATUS_LABEL;
  readonly turnSpan = turnSpan;
  readonly workItemLabel = workItemLabel;
  readonly plural = plural;
  readonly previewUrl = artifactPreviewUrl;
  readonly anchorLabel = anchorLabel;
  readonly referenceKey = referenceKey;

  readonly blocks = this.store.blocks;
  /** Work item id → the item it reruns, from the session's work item list. */
  readonly links = computed(() => rerunLinks(this.store.workItems()));
  readonly blocksRecorded = this.store.blocksRecorded;
  readonly blocksError = this.store.blocksError;
  readonly brief = this.store.frozenBrief;

  plotsOpen = signal(false);
  goalOpen = signal(false);
  /** Open disclosure sections and expanded rows; reset when the selection changes. */
  openSections = signal<Set<Section>>(new Set(['workitem']));
  openRows = signal<Set<string>>(new Set());

  private fragment = toSignal(this.route.fragment, { initialValue: this.route.snapshot.fragment });
  readonly selectedId = computed(() => blockIdFromFragment(this.fragment()));
  readonly selected = computed(() => {
    const id = this.selectedId();
    return id === null ? null : this.blocks().find(b => b.block_id === id) ?? null;
  });
  /** `#gate:<block_id>`: the block whose review gate is selected. */
  readonly selectedGateId = computed(() => gateIdFromFragment(this.fragment()));
  readonly gateBlock = computed(() => {
    const id = this.selectedGateId();
    const block = id === null ? null : this.blocks().find(b => b.block_id === id) ?? null;
    return block && block.work_item_id !== null ? block : null;
  });
  readonly selectedGate = computed(() => {
    const b = this.gateBlock();
    return b ? this.gates().get(b.block_id) ?? null : null;
  });
  /** The block the side panel is about: the selected block, or the selected gate's block. */
  readonly focusBlock = computed(() => this.selected() ?? this.gateBlock());
  /** A linked block (or gate) id that is not in the (loaded) block list. */
  readonly missingId = computed(() => {
    if (this.blocksRecorded() === null) return null;
    const id = this.selectedId();
    if (id !== null) return this.selected() ? null : id;
    const gateId = this.selectedGateId();
    return gateId !== null && !this.gateBlock() ? `gate:${gateId}` : null;
  });

  /** Each block's plots, in artifact_paths order. */
  readonly plots = computed(() => {
    const artifacts = this.store.artifacts();
    return new Map(this.blocks().map(b => [b.block_id, blockPlots(b, artifacts)]));
  });
  readonly plotCount = computed(() => [...this.plots().values()].reduce((n, p) => n + p.length, 0));
  // ── Review gates ──
  /** Work item ids with an evaluator review started from this page and not finished. */
  readonly evaluatorRunning = signal<Set<number>>(new Set());
  /** Block id → its review gate (null for implicit blocks). */
  readonly gates = computed(() => {
    const blocks = this.blocks();
    const details = this.store.workItemDetails();
    const errors = this.store.workItemDetailErrors();
    const qcMode = this.qcMode();
    const running = this.evaluatorRunning();
    return new Map<string, Gate | null>(blocks.map(block => [block.block_id, blockGate({
      block,
      blocks,
      item: block.work_item_id === null ? null : details.get(block.work_item_id) ?? null,
      itemError: block.work_item_id === null ? null : errors.get(block.work_item_id) ?? null,
      qcMode,
      evaluatorRunning: block.work_item_id !== null && running.has(block.work_item_id),
    })]));
  });
  readonly GATE_LABEL = GATE_LABEL;

  // ── Working indicator (ghost card) ──
  /** The agent is working: running, a message awaiting its reply, or code awaiting its result. */
  readonly working = computed(() => {
    const status = this.session()?.status;
    // A session that stopped or errored mid-execution keeps a stale pending
    // code entry (no code_result ever comes): it is not working.
    if (status === 'stopped' || status === 'error') return false;
    return status === 'running' || status === 'initializing' || status === 'recovering' ||
      this.store.waitingForAgent() || this.store.pendingCode().size > 0;
  });
  readonly activity = computed(() => activityLine({
    working: this.working(),
    status: this.session()?.status ?? 'stopped',
    streaming: this.stream.isStreaming(),
    pendingCode: this.working() ? this.store.pendingCode().size : 0,
    mark: this.store.activityMark(),
  }));
  readonly workingSince = this.store.workingSince;
  readonly currentAgent = computed(() => this.session()?.current_agent ?? '');
  readonly queuedCount = computed(() => this.store.messageQueue().length);
  /** The block a message sent from this page is about, while the agent works on it. */
  readonly inFlightBlockId = computed(() => this.working() ? this.store.inFlightBlockId() : null);

  // ── Branch view: the parent's flow above the branch's own blocks ──
  readonly isBranchView = computed(() => {
    const s = this.session();
    return !!s && s.parent_session_id != null && s.forked_from_block_id != null;
  });
  /** The parent's GET /blocks and GET /work-items (for its stage columns): null while loading. */
  readonly parentBlocks = signal<
    { ok: true; blocks: Block[]; items: WorkItemSummary[] } | { ok: false; error: string } | null
  >(null);
  private parentBlocksRequested: string | null = null;
  readonly branchFlow = computed<BranchFlow | null>(() => {
    const s = this.session();
    const parent = this.parentBlocks();
    if (!this.isBranchView() || !s || parent === null) return null;
    if (!parent.ok) return { ok: false, error: parent.error };
    return branchFlow(
      parent.blocks, rerunLinks(parent.items), this.blocks(), this.links(),
      s.parent_session_id!, s.forked_from_block_id!);
  });
  readonly parentRowLabel = computed(() => {
    const p = this.parent();
    const id = this.session()?.parent_session_id;
    return `${p && p.id === id ? p.name : id ? `session ${shortId(id)}` : 'Parent'} (parent)`;
  });
  readonly forkLabel = computed(() => {
    const mode = this.session()?.branch_restore_mode;
    return `branched here · ${mode ? RESTORE_MODE_LABEL[mode] : 'restore mode unknown'}`;
  });
  /** The parent block highlighted for a selected inherited block. */
  readonly highlightedParentId = computed(() => {
    const flow = this.branchFlow();
    const id = this.focusBlock()?.block_id;
    return flow?.ok && id ? flow.parentBlockOf.get(id) ?? null : null;
  });
  /**
   * The step row: one cell per stage (a chain of reruns and retries of one
   * step), or in a branch only its own stages.
   */
  readonly cells = computed<FlowCell[]>(() => {
    if (!this.isBranchView()) return sessionCells(this.blocks(), this.links());
    const flow = this.branchFlow();
    return flow?.ok ? flow.own : ownCellsWithoutParent(this.blocks(), this.links());
  });
  /**
   * Stage key → the version its card shows: the block the panel is about
   * when it is one of the stage's versions, else the latest. Clicking a
   * version in the strip selects it, so `#block:<id>` shows any version.
   */
  readonly shownVersions = computed(() => {
    const focusId = this.focusBlock()?.block_id ?? null;
    return new Map(this.cells().map(c => [c.key, shownVersion(c, focusId)]));
  });
  /** Rows: plots, [parent, fork], steps, then the Paths row. */
  readonly stepRow = computed(() => this.isBranchView() ? 4 : 2);
  readonly lastCell = computed(() => {
    const cells = this.cells();
    return cells.length ? cells[cells.length - 1] : null;
  });
  /** After the last real block; a branch with no own block yet works at its branch point. */
  readonly ghostColumn = computed(() => {
    const last = lastColumn(this.cells());
    const flow = this.branchFlow();
    return last === 0 && flow?.ok ? flow.forkColumn : last + 1;
  });
  readonly gridColumns = computed(() => {
    const flow = this.branchFlow();
    const parentColumns = flow?.ok ? flow.parent.length : 0;
    const steps = Math.max(lastColumn(this.cells()), parentColumns,
      this.working() ? this.ghostColumn() : 0, this.paths.maxColumn());
    return `max-content repeat(${steps}, 184px)`;
  });

  readonly actions = computed(() => {
    const b = this.selected();
    return b ? blockActions(b, this.store.chatItems()) : [];
  });
  /** Action id → its 1-based code row, for the artifacts' "from code N". */
  readonly actionPositions = computed(() => new Map(this.actions().map(a => [a.actionId, a.position])));
  /** Artifacts by the action that last wrote them, for the code cards' plot strips. */
  private readonly artifactsByAction = computed(() => artifactsByAction(this.store.artifacts()));
  private readonly noArtifacts: Artifact[] = [];
  readonly loadedActionCount = computed(() => this.actions().filter(a => a.codeEvent).length);
  readonly artifactEntries = computed(() => {
    const b = this.selected();
    return b ? blockArtifacts(b, this.store.artifacts()) : [];
  });
  readonly selectedPlots = computed(() => {
    const b = this.selected();
    return b ? this.plotsFor(b) : this.noArtifacts;
  });
  readonly workItem = computed(() => {
    const id = this.focusBlock()?.work_item_id;
    return id == null ? null : this.store.workItemDetails().get(id) ?? null;
  });
  readonly workItemSummary = computed(() => {
    const id = this.focusBlock()?.work_item_id;
    return id == null ? null : this.store.workItems().find(w => w.id === id) ?? null;
  });
  readonly workItemError = computed(() => {
    const id = this.focusBlock()?.work_item_id;
    return id == null ? null : this.store.workItemDetailErrors().get(id) ?? null;
  });

  readonly session = this.sessionSvc.currentSession;

  // ── Review feedback and evidence ──
  /** The latest verdict on the selected gate's attempt when it is a reject (evaluator or user). */
  readonly latestReject = computed<WorkItemReview | null>(() => {
    const gate = this.selectedGate();
    const latest = gate ? gate.reviews[gate.reviews.length - 1] : undefined;
    return latest?.verdict === 'reject' ? latest : null;
  });
  /** Work item id → what the evaluator saw in the last review run from this page. */
  readonly evidenceByItem = signal<Map<number, ReviewEvidence>>(new Map());
  readonly evidenceOpen = signal(false);

  /** A branch's parent session (for "inherited from <name>"); null when not a branch or not loaded. */
  readonly parent = signal<Session | null>(null);
  readonly parentError = signal<string | null>(null);
  private parentRequested: string | null = null;
  private branchesRequested: string | null = null;

  /** "inherited from <parent name>", or the source session id if it is not the parent. */
  inheritedLabel(block: Block): string {
    const origin = block.inherited_from;
    if (origin == null) throw new Error(`Block ${block.block_id} is not inherited.`);
    const parent = this.parent();
    const name = parent && parent.id === origin.session_id ? parent.name : `session ${shortId(origin.session_id)}`;
    return `inherited from ${name} (${origin.block_id})`;
  }

  isInherited(block: Block): boolean {
    return block.inherited_from != null;
  }

  /** Why the composer is disabled, worded for the session's actual status. */
  unavailableReason(): string {
    const status = this.session()?.status;
    if (status === 'running' || status === 'initializing' || status === 'recovering') {
      // Busy and not queueable: only interactive sessions past their first
      // turn queue messages.
      return this.session()?.mode === 'interactive'
        ? 'The agent is starting up; you can write once its first turn is done.'
        : 'The session is running automatically; it does not take messages.';
    }
    return `The session is ${status ?? 'not loaded'}; resume it to send a message.`;
  }

  // The session's blueprint: its agents are the ticket owners, its
  // work_item_policy.qc_mode decides when a human review is accepted.
  readonly blueprint = signal<BlueprintContent | null>(null);
  readonly blueprintError = signal<string | null>(null);
  private blueprintRequested: string | null = null;
  /** null while the blueprint is unknown: both Done and In review offer review buttons. */
  readonly qcMode = computed(() => this.blueprint()?.work_item_policy.qc_mode ?? null);
  readonly agentNames = computed(() => {
    const bp = this.blueprint();
    return bp ? Object.keys(bp.agents) : [];
  });

  // "Message the agent about this block".
  readonly composerText = signal('');
  readonly references = signal<BlockReference[]>([]);
  readonly referenceKeys = computed(() => new Set(this.references().map(referenceKey)));
  readonly composerError = signal<string | null>(null);
  readonly composerNotice = signal<string | null>(null);

  // Work item actions.
  readonly reviewForm = signal<'approve' | 'reject' | null>(null);
  readonly reviewText = signal('');
  readonly reviewPending = signal<ReviewAction | null>(null);
  readonly reviewError = signal<string | null>(null);
  readonly reviewNotice = signal<string | null>(null);

  // Open ticket.
  readonly ticketForm = signal<TicketForm | null>(null);
  readonly ticketTitle = signal('');
  readonly ticketBody = signal('');
  readonly ticketOwner = signal('');
  readonly ticketPending = signal(false);
  readonly ticketError = signal<string | null>(null);

  constructor() {
    // The branch count for the Paths row, once per session.
    effect(() => {
      const id = this.session()?.id;
      if (id && id !== this.branchesRequested) {
        this.branchesRequested = id;
        untracked(() => this.paths.refresh());
      }
    });
    // A branch's parent, for the inherited blocks' label.
    effect(() => {
      const parentId = this.session()?.parent_session_id ?? null;
      if (parentId && parentId !== this.parentRequested) {
        this.parentRequested = parentId;
        untracked(() => this.sessionSvc.fetchSession(parentId).subscribe({
          next: p => this.parent.set(p),
          error: err => this.parentError.set(`Could not load the parent session: ${httpErrorMessage(err)}`),
        }));
      }
    });
    // Load the session's blueprint once its name is known.
    effect(() => {
      const name = this.session()?.agent_system;
      if (name && name !== this.blueprintRequested) {
        untracked(() => this.loadBlueprint(name));
      }
    });
    // Every gate needs its work item's reviews: fetch each work item once.
    effect(() => {
      const ids = new Set<number>();
      for (const b of this.blocks()) if (b.work_item_id !== null) ids.add(b.work_item_id);
      untracked(() => { for (const id of ids) this.store.loadWorkItemDetail(id); });
    });
    // A branch's parent flow, for the row above its own blocks.
    effect(() => {
      const parentId = this.isBranchView() ? this.session()!.parent_session_id! : null;
      if (parentId && parentId !== this.parentBlocksRequested) {
        this.parentBlocksRequested = parentId;
        untracked(() => forkJoin([
          this.sessionSvc.getBlocks(parentId), this.sessionSvc.getWorkItems(parentId),
        ]).subscribe({
          next: ([res, items]) => this.parentBlocks.set(res.recorded
            ? { ok: true, blocks: res.blocks, items }
            : { ok: false, error: 'the parent session has no block record' }),
          error: err => this.parentBlocks.set({ ok: false, error: httpErrorMessage(err) }),
        }));
      }
    });
    // Split view: bring the newly selected block into view (the chat pane
    // may have selected it).
    effect(() => {
      const id = this.selectedId();
      if (!this.scrollSelectionIntoView() || id === null) return;
      afterNextRender(() => {
        const root = this.host.nativeElement as HTMLElement;
        const parentId = untracked(this.highlightedParentId);
        const target = root.querySelector(`[data-block-id="${CSS.escape(id)}"]`)
          ?? (parentId ? root.querySelector(`[data-parent-block-id="${CSS.escape(parentId)}"]`) : null);
        // An inherited block whose parent row is unavailable has no cell; the panel still shows it.
        target?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }, { injector: this.injector });
    });
    // A new selection starts with only the work item section open.
    effect(() => {
      this.fragment();
      untracked(() => {
        this.openSections.set(new Set(['workitem']));
        this.openRows.set(new Set());
        this.resetActions();
      });
    });
  }

  select(block: Block): void {
    const fragment = this.selectedId() === block.block_id ? undefined : `block:${block.block_id}`;
    this.navigateFragment(fragment);
  }

  /** Select (or, if selected, deselect) a block's review gate. */
  selectGate(block: Block): void {
    const fragment = this.selectedGateId() === block.block_id ? undefined : `gate:${block.block_id}`;
    this.navigateFragment(fragment);
  }

  openGate(block: Block): void {
    this.navigateFragment(`gate:${block.block_id}`);
  }

  /** The ghost card: select the latest real block, if there is one. */
  selectLatest(): void {
    const last = this.lastCell();
    if (last) this.navigateFragment(`block:${last.block.block_id}`);
  }

  /** A parent block in a branch's top row: open it in the parent's workbench. */
  openParentBlock(block: Block): void {
    const parentId = this.session()?.parent_session_id;
    if (!parentId) throw new Error('This session has no parent.');
    this.router.navigate(['/session', parentId], {
      queryParams: { view: 'workbench' },
      fragment: `block:${block.block_id}`,
    });
  }

  gateGlyph(gate: Gate): string {
    return GATE_GLYPH[gate.state];
  }

  gateFor(block: Block): Gate | null {
    return this.gates().get(block.block_id) ?? null;
  }

  /** The version a stage's card shows (see `shownVersions`). */
  shown(cell: FlowCell): Block {
    const block = this.shownVersions().get(cell.key);
    if (!block) throw new Error(`Stage ${cell.key} has no shown version.`);
    return block;
  }

  /** "v2 ⚠" for a version in a stage's strip. */
  versionText(cell: FlowCell, block: Block): string {
    return versionText(cell, block);
  }

  /** "rerun of blk-0002" / "attempt 2", or null for a first attempt at an original step. */
  rerunLabel(block: Block): string | null {
    return rerunLabel(block, this.links(), this.blocks());
  }

  /** Select a block by id (the evidence list links to blocks that may be earlier versions). */
  navigateTo(blockId: string): void {
    this.navigateFragment(`block:${blockId}`);
  }

  failedEvidenceActions(block: ReviewEvidenceBlock): number {
    return block.actions.filter(a => a.success === false).length;
  }

  missingEvidenceRecords(block: ReviewEvidenceBlock): number {
    return block.actions.filter(a => a.missing_record === true).length;
  }

  /** The cell right after this one holds the next attempt (for the retry line). */
  retryNext(cell: FlowCell, gate: Gate): boolean {
    const next = gate.nextAttempt;
    if (gate.state !== 'changes' || !next) return false;
    return this.cells().some(c => c.block.block_id === next.block_id && c.column === cell.column + 1);
  }

  /** Whether a connector continues past this cell (to a next step, or the ghost). */
  continuesAfter(cell: FlowCell): boolean {
    const last = this.lastCell();
    return last === null || cell.block.block_id !== last.block.block_id || this.working();
  }

  closePanel(): void {
    this.navigateFragment(undefined);
  }

  toggleSection(section: Section): void {
    this.openSections.update(s => toggled(s, section));
  }

  toggleRow(key: string): void {
    this.openRows.update(s => toggled(s, key));
  }

  setRow(key: string, open: boolean): void {
    this.openRows.update(s => {
      const next = new Set(s);
      if (open) next.add(key);
      else next.delete(key);
      return next;
    });
  }

  actionArtifacts(actionId: string): Artifact[] {
    return this.artifactsByAction().get(actionId) ?? this.noArtifacts;
  }

  /** Open the Code section, expand the action's code row and scroll to it. */
  showCode(actionId: string): void {
    this.openSections.update(s => new Set(s).add('code'));
    this.setRow(actionId, true);
    afterNextRender(() => {
      const row = document.getElementById('wb-code-' + actionId);
      if (!row) throw new Error(`Code row for action ${actionId} is not rendered`);
      row.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, { injector: this.injector });
  }

  /** A branch was created from the selected block: refresh the Paths row. */
  onBranchCreated(): void {
    this.paths.refresh();
  }

  plotsFor(block: Block): Artifact[] {
    return this.plots().get(block.block_id) ?? this.noArtifacts;
  }

  // ── Message the agent about this block ──

  toggleReference(ref: BlockReference): void {
    const key = referenceKey(ref);
    this.references.update(refs =>
      refs.some(r => referenceKey(r) === key) ? refs.filter(r => referenceKey(r) !== key) : [...refs, ref]);
  }

  isReferenced(ref: BlockReference): boolean {
    return this.referenceKeys().has(referenceKey(ref));
  }

  sendBlockMessage(block: Block): void {
    this.composerError.set(null);
    this.composerNotice.set(null);
    if (!this.canWrite()) {
      this.composerError.set(`The session cannot take a message right now. ${this.unavailableReason()}`);
      return;
    }
    const queued = !this.canSend();
    if (queued && this.stream.connectionState() !== 'open') {
      this.composerError.set('Not connected to the server, so the message was not queued. Your text is kept.');
      return;
    }
    const content = formatBlockMessage(block, this.references(), this.composerText());
    this.sendMessage.emit({ content, blockId: block.block_id });
    this.composerText.set('');
    this.references.set([]);
    this.composerNotice.set(queued
      ? 'Queued. It is sent when the agent is ready, and shows in the queue above the chat input until then.'
      : 'Sent. It appears in the chat like a typed message.');
  }

  // ── Work item actions ──

  canHumanReview(item: WorkItemDetail): boolean {
    return canHumanReview(item.status, this.qcMode());
  }

  canRunReview(item: WorkItemDetail): boolean {
    return (this.session()?.can_evaluate ?? false) && (item.status === 'In review' || item.status === 'Done');
  }

  openReviewForm(kind: 'approve' | 'reject'): void {
    this.reviewForm.set(this.reviewForm() === kind ? null : kind);
    this.reviewText.set('');
    this.reviewError.set(null);
    this.reviewNotice.set(null);
  }

  /** Open Request changes with the rejecting review's assessment as the (editable) reason. */
  sendBackWithFeedback(review: WorkItemReview): void {
    if (review.verdict !== 'reject') throw new Error('Only a rejecting review can be sent back as feedback.');
    this.reviewForm.set('reject');
    this.reviewText.set(review.assessment.trim());
    this.reviewError.set(null);
    this.reviewNotice.set(null);
  }

  approve(block: Block, item: WorkItemDetail): void {
    const sessionId = this.requireSessionId();
    this.startReview('approve');
    this.sessionSvc.humanReviewWorkItem(sessionId, item.id, {
      verdict: 'approve',
      assessment: this.reviewText().trim(),
    }).subscribe({
      next: () => this.finishReview(block, `Work item #${item.id} approved.`),
      error: err => this.failReview(block, `Approve #${item.id}`, err),
    });
  }

  /**
   * Reject (D.2), then tell the agent with a block-scoped message (E). If the
   * reject fails nothing is sent; if the session became busy meanwhile the
   * reject stands and the unsent message is reported.
   */
  requestChanges(block: Block, item: WorkItemDetail): void {
    const reason = this.reviewText().trim();
    if (!reason) {
      this.reviewError.set('Say what needs to change.');
      return;
    }
    const sessionId = this.requireSessionId();
    this.startReview('reject');
    this.sessionSvc.humanReviewWorkItem(sessionId, item.id, {
      verdict: 'reject',
      assessment: reason,
    }).subscribe({
      next: () => {
        const text = sentBackText(item.id, reason);
        if (!this.canSend()) {
          this.failReview(block, `Request changes on #${item.id}`, new Error(
            `Work item #${item.id} was sent back, but the message to the agent was not sent because the ` +
            `session is busy. Send it from the composer when the agent is idle: "${text}"`));
          return;
        }
        this.sendMessage.emit({ content: formatBlockMessage(block, [], text), blockId: block.block_id });
        this.finishReview(block, `Work item #${item.id} sent back for changes. The agent has been told.`);
      },
      error: err => this.failReview(block, `Request changes on #${item.id}`, err),
    });
  }

  runReview(block: Block, item: WorkItemDetail): void {
    const sessionId = this.requireSessionId();
    this.startReview('evaluator');
    this.reviewForm.set(null);
    this.evaluatorRunning.update(ids => new Set(ids).add(item.id));
    const done = () => this.evaluatorRunning.update(ids => {
      const next = new Set(ids);
      next.delete(item.id);
      return next;
    });
    this.sessionSvc.reviewWorkItem(sessionId, item.id).subscribe({
      next: result => {
        done();
        if (result.evidence) {
          const evidence = result.evidence;
          this.evidenceByItem.update(m => new Map(m).set(item.id, evidence));
        }
        this.finishReview(block, `Evaluator review of #${item.id} recorded: ${result.verdict}.`);
      },
      error: err => {
        done();
        this.failReview(block, `Review of #${item.id}`, err);
      },
    });
  }

  private startReview(action: ReviewAction): void {
    this.reviewPending.set(action);
    this.reviewError.set(null);
    this.reviewNotice.set(null);
  }

  /** Show the outcome in the gate panel, or as a toast if the user has moved off that gate. */
  private finishReview(block: Block, notice: string): void {
    this.reviewPending.set(null);
    if (this.selectedGateId() !== block.block_id) {
      this.toasts.show({ kind: 'success', title: notice, ttlMs: 5000 });
      return;
    }
    this.reviewForm.set(null);
    this.reviewText.set('');
    this.reviewNotice.set(notice);
  }

  private failReview(block: Block, what: string, err: unknown): void {
    this.reviewPending.set(null);
    const message = httpErrorMessage(err);
    if (this.selectedGateId() !== block.block_id) {
      this.toasts.show({ kind: 'error', title: `${what} failed (${block.block_id})`, detail: message, ttlMs: 0 });
      return;
    }
    this.reviewError.set(message);
  }

  // ── Open ticket ──

  openTicket(block: Block, from: { actionId?: string; path?: string } = {}): void {
    const anchor: WorkItemAnchor = {
      block_id: block.block_id,
      action_id: from.actionId ?? null,
      artifact_path: from.path ?? null,
    };
    const origin = from.actionId ? `code:${from.actionId}` : from.path ? `artifact:${from.path}` : 'block';
    if (this.ticketForm()?.origin === origin) {
      this.closeTicket();
      return;
    }
    this.ticketForm.set({ origin, anchor });
    this.ticketTitle.set('');
    this.ticketBody.set('');
    this.ticketError.set(null);
    const current = this.session()?.current_agent ?? '';
    this.ticketOwner.set(this.agentNames().includes(current) ? current : '');
  }

  closeTicket(): void {
    this.ticketForm.set(null);
    this.ticketError.set(null);
  }

  canSubmitTicket(): boolean {
    return !this.ticketPending() && this.ticketTitle().trim() !== ''
      && this.agentNames().includes(this.ticketOwner());
  }

  submitTicket(): void {
    const form = this.ticketForm();
    if (!form) throw new Error('No ticket form is open.');
    const sessionId = this.requireSessionId();
    const owner = this.ticketOwner();
    this.ticketPending.set(true);
    this.ticketError.set(null);
    this.sessionSvc.createWorkItem(sessionId, {
      title: this.ticketTitle().trim(),
      body: this.ticketBody().trim(),
      owner,
      anchor: form.anchor,
    }).subscribe({
      next: item => {
        this.ticketPending.set(false);
        if (this.ticketForm() === form) this.ticketForm.set(null);
        this.toasts.show({
          kind: 'success',
          title: `Ticket #${item.id} opened for ${item.owner}. The agent picks it up on its next turn.`,
          ttlMs: 6000,
        });
      },
      error: err => {
        this.ticketPending.set(false);
        const message = httpErrorMessage(err);
        if (this.ticketForm() !== form) {
          // The form was closed or replaced meanwhile: don't lose the error.
          this.toasts.show({ kind: 'error', title: 'Could not open the ticket', detail: message, ttlMs: 0 });
          return;
        }
        this.ticketError.set(message);
      },
    });
  }

  private loadBlueprint(name: string): void {
    this.blueprintRequested = name;
    this.blueprintError.set(null);
    this.configSvc.getBlueprintContent(name).subscribe({
      next: bp => this.blueprint.set(bp),
      error: err => this.blueprintError.set(`Could not load blueprint "${name}": ${httpErrorMessage(err)}`),
    });
  }

  private resetActions(): void {
    this.composerText.set('');
    this.references.set([]);
    this.composerError.set(null);
    this.composerNotice.set(null);
    this.reviewForm.set(null);
    this.reviewText.set('');
    this.reviewError.set(null);
    this.reviewNotice.set(null);
    this.evidenceOpen.set(false);
    this.ticketForm.set(null);
    this.ticketError.set(null);
  }

  private requireSessionId(): string {
    const id = this.session()?.id;
    if (!id) throw new Error('The workbench has no session.');
    return id;
  }

  private navigateFragment(fragment: string | undefined): void {
    this.router.navigate([], {
      relativeTo: this.route,
      queryParamsHandling: 'preserve',
      fragment,
      replaceUrl: true,
    });
  }
}

function toggled<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value); else next.add(value);
  return next;
}
