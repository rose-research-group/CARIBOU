import { Component, computed, inject, input, model, output, signal } from '@angular/core';
import { Artifact } from '../../../core/models/session.model';
import { PreferencesService } from '../../../core/services/preferences.service';
import { CodeEvent } from '../../../core/state/session-state.model';
import { artifactDownloadUrl, artifactPreviewUrl, formatArtifactSize } from '../../../core/utils/artifacts';
import { ArtifactLightboxComponent } from '../artifact-lightbox/artifact-lightbox';
import { IconComponent } from '../icon/icon';
import { codeLineCount, codeSummary, lastStderrLine } from './code-summary';

/**
 * One executed code block: a collapsed header (agent, status, duration, a
 * one-line summary, line count and, on failure, the last stderr line), the
 * plots the action produced as a thumbnail strip, and, when expanded, the
 * source, output, other produced files and Copy. An attribute component
 * (`<div appCodeCard>`) so the card's element is the host itself and the chat
 * DOM stays flat.
 *
 * Cards start collapsed. `expanded` is a model: unbound (the chat), the card
 * keeps its own state; bound (the workbench), the parent can open it.
 */
@Component({
  selector: '[appCodeCard]',
  standalone: true,
  imports: [IconComponent, ArtifactLightboxComponent],
  templateUrl: './code-card.html',
  styleUrl: './code-card.scss',
  host: { class: 'code-item' },
})
export class CodeCardComponent {
  private prefsSvc = inject(PreferencesService);

  readonly codeEvent = input.required<CodeEvent>();
  /** Artifacts whose `action_id` is this card's action (see artifactsForAction). */
  readonly artifacts = input<Artifact[]>([]);
  readonly expanded = model(false);
  /** The user asked to copy the source; the page owns the clipboard toast. */
  readonly copySource = output<string>();

  readonly lightboxArtifact = signal<Artifact | null>(null);

  readonly summary = computed(() => codeSummary(this.codeEvent().submitted.source));
  readonly lineCount = computed(() => codeLineCount(this.codeEvent().submitted.source));
  readonly errorLine = computed(() => {
    const result = this.codeEvent().result;
    return result && !result.success ? lastStderrLine(result.stderr) : null;
  });
  readonly plots = computed(() => this.artifacts().filter(a => a.type === 'plot'));
  readonly files = computed(() => this.artifacts().filter(a => a.type !== 'plot'));

  readonly previewUrl = artifactPreviewUrl;
  readonly downloadUrl = artifactDownloadUrl;
  readonly formatSize = formatArtifactSize;

  toggle(): void {
    this.expanded.update(open => !open);
  }

  isSlow(): boolean {
    const ms = this.codeEvent().result?.duration_ms ?? 0;
    return ms >= this.prefsSvc.prefs().slowCodeThresholdMs;
  }
}
