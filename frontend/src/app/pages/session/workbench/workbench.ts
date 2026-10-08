import { Component, computed, effect, inject, output, signal, untracked } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router } from '@angular/router';
import { SessionStore } from '../../../core/state/session-store.service';
import { Block } from '../../../core/models/block.model';
import { Artifact } from '../../../core/models/session.model';
import { artifactPreviewUrl } from '../../../core/utils/artifacts';
import { CodeCardComponent } from '../../../shared/components/code-card/code-card';
import { ArtifactCardComponent } from '../../../shared/components/artifact-card/artifact-card';
import { IconComponent } from '../../../shared/components/icon/icon';
import {
  STATUS_GLYPH, STATUS_LABEL, blockActions, blockArtifacts, blockIdFromFragment,
  firstPlot, plural, turnSpan, workItemLabel,
} from './block-matching';

type Section = 'workitem' | 'code' | 'artifacts';

/**
 * Read-only workbench view of a session (workbench step 3): the Goal, one
 * card per block in `index` order, a foldable Plots row, and a side panel for
 * the selected block. Everything comes from the page's SessionStore; the
 * selection is the URL fragment (`#block:<block_id>`), so it deep-links.
 */
@Component({
  selector: 'app-workbench',
  standalone: true,
  imports: [CodeCardComponent, ArtifactCardComponent, IconComponent],
  templateUrl: './workbench.html',
  styleUrl: './workbench.scss',
})
export class WorkbenchComponent {
  private store = inject(SessionStore);
  private route = inject(ActivatedRoute);
  private router = inject(Router);

  /** A code card's Copy button; the page owns the clipboard and its toast. */
  readonly copySource = output<string>();

  readonly STATUS_GLYPH = STATUS_GLYPH;
  readonly STATUS_LABEL = STATUS_LABEL;
  readonly turnSpan = turnSpan;
  readonly workItemLabel = workItemLabel;
  readonly plural = plural;
  readonly previewUrl = artifactPreviewUrl;

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

  constructor() {
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
