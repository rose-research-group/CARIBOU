import { Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router } from '@angular/router';
import { SessionStore } from '../../../core/state/session-store.service';
import { Block } from '../../../core/models/block.model';
import { Artifact, WorkItemAnchor, WorkItemDetail } from '../../../core/models/session.model';
import { BlueprintContent } from '../../../core/models/blueprint.model';
import { SessionService } from '../../../core/services/session.service';
import { ConfigService } from '../../../core/services/config.service';
import { ToastService } from '../../../core/services/toast.service';
import { artifactPreviewUrl } from '../../../core/utils/artifacts';
import { CodeCardComponent } from '../../../shared/components/code-card/code-card';
import { ArtifactCardComponent } from '../../../shared/components/artifact-card/artifact-card';
import { IconComponent } from '../../../shared/components/icon/icon';
import {
  STATUS_GLYPH, STATUS_LABEL, blockActions, blockArtifacts, blockIdFromFragment,
  firstPlot, plural, turnSpan, workItemLabel,
} from './block-matching';
import {
  BlockReference, anchorLabel, canHumanReview, formatBlockMessage, httpErrorMessage,
  referenceKey, sentBackText,
} from './block-actions';

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
@Component({
  selector: 'app-workbench',
  standalone: true,
  imports: [CodeCardComponent, ArtifactCardComponent, IconComponent, NgTemplateOutlet],
  templateUrl: './workbench.html',
  styleUrl: './workbench.scss',
})
export class WorkbenchComponent {
  private store = inject(SessionStore);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private sessionSvc = inject(SessionService);
  private configSvc = inject(ConfigService);
  private toasts = inject(ToastService);

  /** A code card's Copy button; the page owns the clipboard and its toast. */
  readonly copySource = output<string>();
  /** Whether the session takes a user message now (the chat input's condition). */
  readonly canSend = input.required<boolean>();
  /** A block-scoped message; the page sends it through the chat's send path. */
  readonly sendMessage = output<string>();

  readonly STATUS_GLYPH = STATUS_GLYPH;
  readonly STATUS_LABEL = STATUS_LABEL;
  readonly turnSpan = turnSpan;
  readonly workItemLabel = workItemLabel;
  readonly plural = plural;
  readonly previewUrl = artifactPreviewUrl;
  readonly anchorLabel = anchorLabel;
  readonly referenceKey = referenceKey;

  readonly blocks = this.store.blocks;
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
  /** A linked block id that is not in the (loaded) block list. */
  readonly missingId = computed(() => {
    const id = this.selectedId();
    return id !== null && this.blocksRecorded() !== null && !this.selected() ? id : null;
  });

  readonly plots = computed(() => {
    const artifacts = this.store.artifacts();
    return new Map(this.blocks().map(b => [b.block_id, firstPlot(b, artifacts)]));
  });
  readonly plotCount = computed(() => [...this.plots().values()].filter(p => p !== null).length);
  readonly gridColumns = computed(() => `max-content repeat(${this.blocks().length}, 184px)`);

  readonly actions = computed(() => {
    const b = this.selected();
    return b ? blockActions(b, this.store.chatItems()) : [];
  });
  readonly loadedActionCount = computed(() => this.actions().filter(a => a.codeEvent).length);
  readonly artifactEntries = computed(() => {
    const b = this.selected();
    return b ? blockArtifacts(b, this.store.artifacts()) : [];
  });
  readonly selectedPlot = computed(() => {
    const b = this.selected();
    return b ? this.plots().get(b.block_id) ?? null : null;
  });
  readonly workItem = computed(() => {
    const id = this.selected()?.work_item_id;
    return id == null ? null : this.store.workItemDetails().get(id) ?? null;
  });
  readonly workItemSummary = computed(() => {
    const id = this.selected()?.work_item_id;
    return id == null ? null : this.store.workItems().find(w => w.id === id) ?? null;
  });
  readonly workItemError = computed(() => {
    const id = this.selected()?.work_item_id;
    return id == null ? null : this.store.workItemDetailErrors().get(id) ?? null;
  });

  readonly session = this.sessionSvc.currentSession;

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
    // Load the session's blueprint once its name is known.
    effect(() => {
      const name = this.session()?.agent_system;
      if (name && name !== this.blueprintRequested) {
        untracked(() => this.loadBlueprint(name));
      }
    });
    // Fetch the selected block's work item (with reviews) on first selection.
    effect(() => {
      const id = this.selected()?.work_item_id;
      if (id != null) untracked(() => this.store.loadWorkItemDetail(id));
    });
    // A new selection starts with only the work item section open.
    effect(() => {
      this.selectedId();
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

  closePanel(): void {
    this.navigateFragment(undefined);
  }

  toggleSection(section: Section): void {
    this.openSections.update(s => toggled(s, section));
  }

  toggleRow(key: string): void {
    this.openRows.update(s => toggled(s, key));
  }

  plotFor(block: Block): Artifact | null {
    return this.plots().get(block.block_id) ?? null;
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
    if (!this.canSend()) {
      this.composerError.set('The session cannot take a message right now. Wait until the agent is idle.');
      return;
    }
    const content = formatBlockMessage(block, this.references(), this.composerText());
    this.sendMessage.emit(content);
    this.composerText.set('');
    this.references.set([]);
    this.composerNotice.set('Sent. It appears in the chat like a typed message.');
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
        this.sendMessage.emit(formatBlockMessage(block, [], text));
        this.finishReview(block, `Work item #${item.id} sent back for changes. The agent has been told.`);
      },
      error: err => this.failReview(block, `Request changes on #${item.id}`, err),
    });
  }

  runReview(block: Block, item: WorkItemDetail): void {
    const sessionId = this.requireSessionId();
    this.startReview('evaluator');
    this.reviewForm.set(null);
    this.sessionSvc.reviewWorkItem(sessionId, item.id).subscribe({
      next: result => this.finishReview(block, `Evaluator review of #${item.id} recorded: ${result.verdict}.`),
      error: err => this.failReview(block, `Review of #${item.id}`, err),
    });
  }

  private startReview(action: ReviewAction): void {
    this.reviewPending.set(action);
    this.reviewError.set(null);
    this.reviewNotice.set(null);
  }

  /** Show the outcome in the panel, or as a toast if the user has moved to another block. */
  private finishReview(block: Block, notice: string): void {
    this.reviewPending.set(null);
    if (this.selectedId() !== block.block_id) {
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
    if (this.selectedId() !== block.block_id) {
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
