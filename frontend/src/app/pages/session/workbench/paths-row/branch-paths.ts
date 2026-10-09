import { DestroyRef, Injectable, computed, inject, signal } from '@angular/core';
import { forkJoin, of, catchError, map } from 'rxjs';
import { Block } from '../../../../core/models/block.model';
import { BranchSummary, WorkItemSummary } from '../../../../core/models/session.model';
import { SessionService } from '../../../../core/services/session.service';
import { SessionStore } from '../../../../core/state/session-store.service';
import { httpErrorMessage } from '../block-actions';
import { LaneLayout, laneLayout } from '../branching';
import { columnByBlock, sessionCells } from '../flow-layout';
import { rerunLinks } from '../stages';

/**
 * A lane's own GET /blocks and GET /work-items (the items give the rerun
 * links that group its blocks into stages): in flight, loaded, or failed
 * with the reason.
 */
export type LaneBlocks =
  | { kind: 'loading' }
  | { kind: 'loaded'; blocks: Block[]; items: WorkItemSummary[] }
  | { kind: 'error'; message: string };

const POLL_MS = 5000;

/**
 * The workbench's branch lanes: GET /branches for the session, then each
 * branch's GET /blocks, laid out under the session's step columns. Provided
 * by the workbench so its grid can widen to the longest lane. While the
 * Paths row is open and a branch is still starting, restoring or running it
 * is refetched every few seconds.
 */
@Injectable()
export class BranchPaths {
  private sessionSvc = inject(SessionService);
  private store = inject(SessionStore);

  readonly open = signal(false);
  readonly branches = signal<BranchSummary[] | null>(null);
  readonly branchesError = signal<string | null>(null);
  readonly laneBlocks = signal<Map<string, LaneBlocks>>(new Map());

  private generation = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private readonly sessionId = computed(() => this.sessionSvc.currentSession()?.id ?? null);

  /** Branch id → its layout under the current stage columns (only for loaded lanes). */
  readonly layouts = computed(() => {
    const id = this.sessionId();
    const columns = columnByBlock(sessionCells(this.store.blocks(), rerunLinks(this.store.workItems())));
    const out = new Map<string, LaneLayout>();
    if (id === null) return out;
    for (const branch of this.branches() ?? []) {
      const lane = this.laneBlocks().get(branch.session_id);
      if (lane?.kind === 'loaded') {
        out.set(branch.session_id, laneLayout(
          columns, lane.blocks, rerunLinks(lane.items), id, branch.forked_from_block_id));
      }
    }
    return out;
  });

  /** The furthest step column any lane reaches (0 with none). */
  readonly maxColumn = computed(() => {
    if (!this.open()) return 0;
    let max = 0;
    for (const layout of this.layouts().values()) {
      if (layout.ok) max = Math.max(max, layout.lastColumn);
    }
    return max;
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => this.stopPolling());
  }

  toggle(): void {
    this.open.update(v => !v);
    if (this.open()) {
      this.refresh();
      this.timer = setInterval(() => {
        if (this.anyActive()) this.refresh();
      }, POLL_MS);
    } else {
      this.stopPolling();
    }
  }

  /** Fetch the branch list; with the row open, also each branch's blocks. */
  refresh(): void {
    const sessionId = this.sessionId();
    if (sessionId === null) throw new Error('The workbench has no session.');
    const generation = ++this.generation;
    const withLanes = this.open();
    this.sessionSvc.getBranches(sessionId).subscribe({
      next: branches => {
        if (generation !== this.generation) return; // superseded by a newer refresh
        this.branches.set(branches);
        this.branchesError.set(null);
        if (withLanes) this.loadLanes(branches, generation);
      },
      error: err => {
        if (generation !== this.generation) return;
        this.branchesError.set(httpErrorMessage(err));
      },
    });
  }

  private loadLanes(branches: BranchSummary[], generation: number): void {
    if (branches.length === 0) return;
    // A lane already shown keeps its blocks while the refetch is in flight.
    this.laneBlocks.update(prev => {
      const next = new Map(prev);
      for (const b of branches) if (!next.has(b.session_id)) next.set(b.session_id, { kind: 'loading' });
      return next;
    });
    // Each lane's failure is its own visible state; it doesn't fail the others.
    forkJoin(branches.map(b => forkJoin([
      this.sessionSvc.getBlocks(b.session_id), this.sessionSvc.getWorkItems(b.session_id),
    ]).pipe(
      map(([res, items]): [string, LaneBlocks] => [b.session_id, { kind: 'loaded', blocks: res.blocks, items }]),
      catchError(err => of<[string, LaneBlocks]>([b.session_id, { kind: 'error', message: httpErrorMessage(err) }])),
    ))).subscribe(results => {
      if (generation !== this.generation) return;
      this.laneBlocks.set(new Map(results));
    });
  }

  private anyActive(): boolean {
    return (this.branches() ?? []).some(b =>
      b.status === 'initializing' || b.status === 'running' || b.status === 'recovering'
      || b.recovery_status === 'awaiting_checkpoint' || b.recovery_status === 'recovering');
  }

  private stopPolling(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
  }
}
