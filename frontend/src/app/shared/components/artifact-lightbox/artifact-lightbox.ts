import { Component, ElementRef, HostListener, afterNextRender, computed, input, output, viewChild } from '@angular/core';
import { Artifact } from '../../../core/models/session.model';
import { artifactDownloadUrl, artifactPreviewUrl } from '../../../core/utils/artifacts';
import { IconComponent } from '../icon/icon';

/**
 * Full-screen preview of one plot artifact with a download link. The parent
 * renders it while open and removes it on `closed` (overlay click, the close
 * button or Escape). A parent showing several plots can pass `position`,
 * `hasPrev` and `hasNext` and handle `prev` / `next` (buttons and arrow keys).
 */
@Component({
  selector: 'app-artifact-lightbox',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './artifact-lightbox.html',
  styleUrl: './artifact-lightbox.scss',
})
export class ArtifactLightboxComponent {
  readonly artifact = input.required<Artifact>();
  readonly closed = output<void>();
  /** "3 / 6" when the plot is one of several; null for a single plot. */
  readonly position = input<string | null>(null);
  readonly hasPrev = input(false);
  readonly hasNext = input(false);
  readonly prev = output<void>();
  readonly next = output<void>();

  readonly previewUrl = computed(() => artifactPreviewUrl(this.artifact()));
  readonly downloadUrl = computed(() => artifactDownloadUrl(this.artifact()));

  private readonly closeButton = viewChild.required<ElementRef<HTMLButtonElement>>('closeButton');

  constructor() {
    // The dialog is modal: move keyboard focus into it when it opens.
    afterNextRender(() => this.closeButton().nativeElement.focus());
  }

  @HostListener('document:keydown.escape')
  onEscape(): void { this.closed.emit(); }

  @HostListener('document:keydown.arrowleft')
  onArrowLeft(): void { if (this.hasPrev()) this.prev.emit(); }

  @HostListener('document:keydown.arrowright')
  onArrowRight(): void { if (this.hasNext()) this.next.emit(); }
}
