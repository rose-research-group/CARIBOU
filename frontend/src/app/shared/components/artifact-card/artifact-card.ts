import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Artifact } from '../../../core/models/session.model';
import { artifactDownloadUrl, artifactPreviewUrl, formatArtifactSize } from '../../../core/utils/artifacts';
import { ArtifactLightboxComponent } from '../artifact-lightbox/artifact-lightbox';
import { TooltipDirective } from '../../directives/tooltip.directive';
import { IconComponent } from '../icon/icon';

@Component({
  selector: 'app-artifact-card',
  standalone: true,
  imports: [CommonModule, TooltipDirective, IconComponent, ArtifactLightboxComponent],
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

  get downloadUrl(): string {
    return artifactDownloadUrl(this.artifact);
  }

  get previewUrl(): string {
    return artifactPreviewUrl(this.artifact);
  }

  formatSize(bytes: number): string {
    return formatArtifactSize(bytes);
  }
}
