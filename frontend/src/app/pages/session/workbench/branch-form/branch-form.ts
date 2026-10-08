import { Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core';
import { Router } from '@angular/router';
import { Block } from '../../../../core/models/block.model';
import { BranchRestoreMode, Session, SessionStatus } from '../../../../core/models/session.model';
import { SessionService } from '../../../../core/services/session.service';
import { ToastService } from '../../../../core/services/toast.service';
import { IconComponent } from '../../../../shared/components/icon/icon';
import { httpErrorDetail } from '../block-actions';
import {
  BRANCH_INSTRUCTION_MAX, BranchFormValue, branchFormError, branchUnavailableReason, buildBranchRequest,
  defaultRestoreMode, restoreModeOptions,
} from '../branching';

/**
 * "Branch from this block": a new session that starts from the block's
 * entry checkpoint, restored with the chosen mode, whose first user turn is
 * the instruction. Modes the block's entry cannot support are disabled with
 * the reason; the whole form is disabled while the source is not idle,
 * stopped or in error. Server errors are shown as the server's `detail`.
 */
@Component({
  selector: 'app-branch-form',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './branch-form.html',
  styleUrl: './branch-form.scss',
})
export class BranchFormComponent {
  private sessionSvc = inject(SessionService);
  private toasts = inject(ToastService);
  private router = inject(Router);

  readonly block = input.required<Block>();
  readonly sessionId = input.required<string>();
  readonly sourceStatus = input.required<SessionStatus | null>();
  /** The new child session (the Paths row refreshes on it). */
  readonly created = output<Session>();

  readonly INSTRUCTION_MAX = BRANCH_INSTRUCTION_MAX;

  readonly open = signal(false);
  readonly instruction = signal('');
  readonly name = signal('');
  readonly mode = signal<BranchRestoreMode | null>(null);
  readonly acknowledge = signal(false);
  readonly pending = signal(false);
  readonly error = signal<string | null>(null);

  private readonly blockId = computed(() => this.block().block_id);
  readonly options = computed(() => restoreModeOptions(this.block()));
  readonly unavailable = computed(() => branchUnavailableReason(this.block(), this.sourceStatus()));
  readonly selectedOption = computed(() => this.options().find(o => o.mode === this.mode()) ?? null);
  private readonly value = computed<BranchFormValue>(() => ({
    instruction: this.instruction(),
    mode: this.mode(),
    name: this.name(),
    acknowledgeUnverified: this.acknowledge(),
  }));
  /** Why Create is disabled (shown under the button), or null. */
  readonly formError = computed(() => this.unavailable() ?? branchFormError(this.value(), this.options()));

  constructor() {
    // A new block starts with a closed, empty form and its own default mode.
    // Tracks the id only: a block_changed for the same block keeps the draft.
    effect(() => {
      this.blockId();
      untracked(() => this.reset());
    });
  }

  toggle(): void {
    this.open.update(v => !v);
    this.error.set(null);
  }

  chooseMode(mode: BranchRestoreMode): void {
    this.mode.set(mode);
    this.acknowledge.set(false);
  }

  submit(): void {
    const unavailable = this.unavailable();
    if (unavailable !== null) {
      this.error.set(unavailable);
      return;
    }
    const block = this.block();
    const request = buildBranchRequest(this.value(), this.options());
    this.pending.set(true);
    this.error.set(null);
    this.sessionSvc.branchFromBlock(this.sessionId(), block.block_id, request).subscribe({
      next: child => {
        this.pending.set(false);
        this.reset();
        this.created.emit(child);
        this.toasts.show({
          kind: 'success',
          title: `Branch "${child.name}" created from ${block.block_id}`,
          detail: 'It restores the checkpoint, then starts on your instruction.',
          action: { label: 'Open branch', run: () => this.openChild(child.id) },
          ttlMs: 0,
        });
      },
      error: err => {
        this.pending.set(false);
        this.error.set(httpErrorDetail(err).text);
      },
    });
  }

  private openChild(id: string): void {
    void this.router.navigate(['/session', id], { queryParams: { view: 'workbench' } });
  }

  private reset(): void {
    this.open.set(false);
    this.instruction.set('');
    this.name.set('');
    this.mode.set(defaultRestoreMode(this.options()));
    this.acknowledge.set(false);
    this.error.set(null);
  }
}
