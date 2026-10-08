import {
  Component, ElementRef, Injector, afterNextRender, computed, effect, inject, input, signal, untracked, viewChild,
} from '@angular/core';
import { Artifact } from '../../../core/models/session.model';
import { artifactPreviewUrl } from '../../../core/utils/artifacts';
import { ArtifactLightboxComponent } from '../artifact-lightbox/artifact-lightbox';
import { IconComponent } from '../icon/icon';

/**
 * A row of plots, one at a time: prev/next buttons, a "3 / 6" position,
 * arrow keys when focused, and native horizontal scrolling with CSS
 * scroll-snap (so touch swipes work). Clicking the plot opens the shared
 * lightbox, which can step through the same plots.
 *
 * The position survives updates to `plots` (clamped if the list shrinks) and
 * resets to the first plot when `resetKey` changes (a different block).
 */
@Component({
  selector: 'app-plot-carousel',
  standalone: true,
  imports: [IconComponent, ArtifactLightboxComponent],
  templateUrl: './plot-carousel.html',
  styleUrl: './plot-carousel.scss',
})
export class PlotCarouselComponent {
  private injector = inject(Injector);

  /** The plots, in display order. Must not be empty. */
  readonly plots = input.required<Artifact[]>();
  /** Identifies what the plots belong to; a new key starts at the first plot. */
  readonly resetKey = input.required<string>();
  /** Small slides for the workbench's plots row. */
  readonly compact = input(false);
  /** Names the carousel for assistive technology, e.g. "Plots from QC". */
  readonly label = input('Plots');

  readonly previewUrl = artifactPreviewUrl;

  private readonly index = signal(0);
  /** The shown position, clamped to the current list. */
  readonly current = computed(() => Math.min(this.index(), Math.max(0, this.plots().length - 1)));
  readonly plot = computed(() => {
    const plots = this.plots();
    if (plots.length === 0) throw new Error('PlotCarousel needs at least one plot.');
    return plots[this.current()];
  });
  readonly multiple = computed(() => this.plots().length > 1);
  readonly position = computed(() => `${this.current() + 1} / ${this.plots().length}`);
  readonly lightboxOpen = signal(false);

  private readonly track = viewChild<ElementRef<HTMLElement>>('track');

  constructor() {
    effect(() => {
      this.resetKey();
      untracked(() => {
        this.index.set(0);
        this.lightboxOpen.set(false);
      });
    });
    // Keep the stored index in range when the list shrinks.
    effect(() => {
      const clamped = this.current();
      if (untracked(this.index) !== clamped) this.index.set(clamped);
    });
    // Keep the scroll position on the current slide (after the list changes,
    // or when the index changes from a button, key or the lightbox).
    effect(() => {
      const i = this.current();
      this.plots();
      afterNextRender(() => this.scrollTo(i, false), { injector: this.injector });
    });
  }

  prev(): void { this.go(this.current() - 1); }
  next(): void { this.go(this.current() + 1); }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowLeft') { this.prev(); event.preventDefault(); }
    else if (event.key === 'ArrowRight') { this.next(); event.preventDefault(); }
  }

  /** A user scroll or swipe: follow it to the slide that is mostly visible. */
  onScroll(): void {
    const el = this.track()?.nativeElement;
    if (!el || el.clientWidth === 0) return;
    const i = Math.round(el.scrollLeft / el.clientWidth);
    if (i !== this.index() && i >= 0 && i < this.plots().length) this.index.set(i);
  }

  private go(i: number): void {
    const n = this.plots().length;
    if (i < 0 || i >= n) return;
    this.index.set(i);
    this.scrollTo(i, true);
  }

  private scrollTo(i: number, animate: boolean): void {
    const el = this.track()?.nativeElement;
    if (!el) return;
    const left = i * el.clientWidth;
    // Already there (or a swipe is settling onto it): leave the scroll alone.
    if (Math.abs(el.scrollLeft - left) < el.clientWidth / 2) return;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    el.scrollTo({ left, behavior: animate && !reduced ? 'smooth' : 'auto' });
  }
}
