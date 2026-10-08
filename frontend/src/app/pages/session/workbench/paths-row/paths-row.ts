import { Component, inject, input } from '@angular/core';
import { Router } from '@angular/router';
import { BranchSummary } from '../../../../core/models/session.model';
import { STATUS_GLYPH, STATUS_LABEL } from '../block-matching';
import { RECOVERY_STATUS_LABEL, RESTORE_MODE_LABEL } from '../branching';
import { BranchPaths } from './branch-paths';

/**
 * The workbench's foldable "Paths" row: one lane per branch of this
 * session, under the same step columns. A lane shows the parent blocks it
 * inherited (dimmed) and its own blocks from the branch point on, with the
 * branch's status, restore status, mode and instruction. Clicking a lane
 * opens that branch's workbench.
 *
 * The host is `display: contents`, so the cells are items of the
 * workbench's grid; `firstRow` is the grid row of the Paths label.
 */
@Component({
  selector: 'app-paths-row',
  standalone: true,
  templateUrl: './paths-row.html',
  styleUrl: './paths-row.scss',
})
export class PathsRowComponent {
  readonly paths = inject(BranchPaths);
  private router = inject(Router);

  readonly firstRow = input.required<number>();

  readonly STATUS_GLYPH = STATUS_GLYPH;
  readonly STATUS_LABEL = STATUS_LABEL;
  readonly RECOVERY_STATUS_LABEL = RECOVERY_STATUS_LABEL;
  readonly RESTORE_MODE_LABEL = RESTORE_MODE_LABEL;

  laneError(sessionId: string): string {
    const lane = this.paths.laneBlocks().get(sessionId);
    if (lane?.kind !== 'error') throw new Error(`Lane ${sessionId} has no error.`);
    return lane.message;
  }

  openBranch(branch: BranchSummary, blockId?: string): void {
    void this.router.navigate(['/session', branch.session_id], {
      queryParams: { view: 'workbench' },
      fragment: blockId ? `block:${blockId}` : undefined,
    });
  }
}
