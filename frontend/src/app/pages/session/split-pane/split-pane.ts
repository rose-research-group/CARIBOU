import { Component, ElementRef, computed, inject, input, signal, viewChild } from '@angular/core';
import { ToastService } from '../../../core/services/toast.service';
import {
  SPLIT_DEFAULT, SPLIT_MAX, SPLIT_MIN, SPLIT_STEP, SPLIT_STORAGE_KEY, clampSplit, parseStoredSplit,
} from './split-ratio';

/**
 * Two panes side by side ([paneStart] and [paneEnd]) with a draggable,
 * keyboard-operable divider, and an optional overlay drawer ([paneDrawer])
 * on the right. The ratio is a per-viewer preference kept in localStorage;
 * if storage is unavailable the split starts at 50/50 and a toast says why.
 */
@Component({
  selector: 'app-split-pane',
  standalone: true,
  templateUrl: './split-pane.html',
  styleUrl: './split-pane.scss',
})
export class SplitPaneComponent {
  private toasts = inject(ToastService);

  readonly drawerOpen = input(false);

  readonly ratio = signal(this.loadRatio());
  readonly columns = computed(() => `minmax(0, ${this.ratio()}fr) 6px minmax(0, ${1 - this.ratio()}fr)`);
  readonly percent = computed(() => Math.round(this.ratio() * 100));
  readonly min = Math.round(SPLIT_MIN * 100);
  readonly max = Math.round(SPLIT_MAX * 100);
  readonly dragging = signal(false);

  private readonly root = viewChild.required<ElementRef<HTMLElement>>('root');

  onPointerDown(event: PointerEvent): void {
    if (event.button !== 0) return;
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
    this.dragging.set(true);
    event.preventDefault();
  }

  onPointerMove(event: PointerEvent): void {
    if (!this.dragging()) return;
    const rect = this.root().nativeElement.getBoundingClientRect();
    if (rect.width === 0) return;
    this.ratio.set(clampSplit((event.clientX - rect.left) / rect.width));
  }

  onPointerUp(event: PointerEvent): void {
    if (!this.dragging()) return;
    (event.target as HTMLElement).releasePointerCapture(event.pointerId);
    this.dragging.set(false);
    this.saveRatio();
  }

  onKeydown(event: KeyboardEvent): void {
    const steps: Record<string, number> = {
      ArrowLeft: this.ratio() - SPLIT_STEP,
      ArrowRight: this.ratio() + SPLIT_STEP,
      Home: SPLIT_MIN,
      End: SPLIT_MAX,
    };
    const next = steps[event.key];
    if (next === undefined) return;
    event.preventDefault();
    this.ratio.set(clampSplit(next));
    this.saveRatio();
  }

  private loadRatio(): number {
    try {
      return parseStoredSplit(localStorage.getItem(SPLIT_STORAGE_KEY)) ?? SPLIT_DEFAULT;
    } catch (err) {
      this.storageFailed(err);
      return SPLIT_DEFAULT;
    }
  }

  private saveRatio(): void {
    try {
      localStorage.setItem(SPLIT_STORAGE_KEY, String(this.ratio()));
    } catch (err) {
      this.storageFailed(err);
    }
  }

  private storageFailed(err: unknown): void {
    this.toasts.show({
      kind: 'warn',
      title: 'Split position not saved',
      detail: `Browser storage is unavailable (${err instanceof Error ? err.message : String(err)}). ` +
        'The split starts at 50/50 each time.',
      ttlMs: 6000,
    });
  }
}
