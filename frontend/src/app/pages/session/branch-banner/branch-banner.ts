import { Component, computed, effect, inject, input, signal, untracked } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Session } from '../../../core/models/session.model';
import { SessionService } from '../../../core/services/session.service';
import { IconComponent } from '../../../shared/components/icon/icon';
import { httpErrorMessage } from '../workbench/block-actions';
import {
  RECOVERY_STATUS_LABEL, RESTORE_MODE_LABEL, restorePercent, restoreProgressText, shortId,
} from '../workbench/branching';

/**
 * Shown above both the chat and the workbench of a branch: what it was
 * branched from (with a link back to the parent) and how its restore is
 * going, from the session's recovery_* fields.
 */
@Component({
  selector: 'app-branch-banner',
  standalone: true,
  imports: [RouterLink, IconComponent],
  templateUrl: './branch-banner.html',
  styleUrl: './branch-banner.scss',
})
export class BranchBannerComponent {
  private sessionSvc = inject(SessionService);

  readonly session = input.required<Session>();

  readonly parent = signal<Session | null>(null);
  readonly parentError = signal<string | null>(null);
  private requested: string | null = null;

  readonly RESTORE_MODE_LABEL = RESTORE_MODE_LABEL;
  readonly RECOVERY_STATUS_LABEL = RECOVERY_STATUS_LABEL;
  readonly restoreProgressText = restoreProgressText;
  readonly restorePercent = restorePercent;

  readonly parentLabel = computed(() => {
    const id = this.session().parent_session_id;
    if (!id) return null;
    return this.parent()?.name || `session ${shortId(id)}`;
  });
  readonly restoring = computed(() => {
    const st = this.session().recovery_status;
    return st === 'awaiting_checkpoint' || st === 'recovering';
  });
  readonly restoreFailed = computed(() => {
    const st = this.session().recovery_status;
    return st === 'failed' || st === 'partial';
  });

  constructor() {
    effect(() => {
      const id = this.session().parent_session_id;
      if (id && id !== this.requested) {
        this.requested = id;
        untracked(() => this.sessionSvc.fetchSession(id).subscribe({
          next: p => this.parent.set(p),
          error: err => this.parentError.set(httpErrorMessage(err)),
        }));
      }
    });
  }
}
