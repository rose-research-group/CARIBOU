import { Component, Input, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Artifact } from '../../../core/models/session.model';
import { TooltipDirective } from '../../directives/tooltip.directive';
import { IconComponent } from '../icon/icon';

@Component({
  selector: 'app-artifact-card',
  standalone: true,
  imports: [CommonModule, TooltipDirective, IconComponent],
  templateUrl: './artifact-card.html',
  styleUrl: './artifact-card.scss',
})
export class ArtifactCardComponent {
  @Input() artifact!: Artifact;
  lightboxOpen = false;

  get isPlot(): boolean {
    return this.artifact.type === 'plot';
  }

  openLightbox(): void  { this.lightboxOpen = true; }
  closeLightbox(): void { this.lightboxOpen = false; }

  @HostListener('document:keydown.escape')
  onEscape(): void { this.lightboxOpen = false; }

  get downloadUrl(): string {
    const base = document.baseURI.replace(/\/$/, '');
    return `${base}/api/sessions/${this.artifact.session_id}/artifacts/${this.artifact.id}/download`;
  }

  /** Preview src; mtime_ns changes when the same path is overwritten, so the
   *  browser does not keep showing the cached image of the old file. */
  get previewUrl(): string {
    return `${this.downloadUrl}?v=${this.artifact.mtime_ns}`;
  }

  formatSize(bytes: number): string {
    if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' KB';
  }
}
