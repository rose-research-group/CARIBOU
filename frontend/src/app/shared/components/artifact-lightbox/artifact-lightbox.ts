import { Component, ElementRef, HostListener, afterNextRender, computed, input, output, viewChild } from '@angular/core';
import { Artifact } from '../../../core/models/session.model';
import { artifactDownloadUrl, artifactPreviewUrl } from '../../../core/utils/artifacts';
import { IconComponent } from '../icon/icon';

/**
 * Full-screen preview of one plot artifact with a download link. The parent
 * renders it while open and removes it on `closed` (overlay click, the close
 * button or Escape).
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

  readonly previewUrl = computed(() => artifactPreviewUrl(this.artifact()));
  readonly downloadUrl = computed(() => artifactDownloadUrl(this.artifact()));

  private readonly closeButton = viewChild.required<ElementRef<HTMLButtonElement>>('closeButton');

  constructor() {
    // The dialog is modal: move keyboard focus into it when it opens.
    afterNextRender(() => this.closeButton().nativeElement.focus());
  }

  @HostListener('document:keydown.escape')
  onEscape(): void { this.closed.emit(); }
}
