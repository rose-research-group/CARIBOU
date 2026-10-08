import { Component, Input, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Artifact } from '../../../core/models/session.model';
import { artifactDownloadUrl, artifactPreviewUrl } from '../../../core/utils/artifacts';
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
    return artifactDownloadUrl(this.artifact);
  }

  get previewUrl(): string {
    return artifactPreviewUrl(this.artifact);
  }

  formatSize(bytes: number): string {
    if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' KB';
  }
}
