import { Component, inject, input, output } from '@angular/core';
import { PreferencesService } from '../../../core/services/preferences.service';
import { CodeEvent } from '../../../core/state/session-state.model';
import { IconComponent } from '../icon/icon';

/**
 * One executed code block in the chat: source, run status, duration and
 * output. An attribute component (`<div appCodeCard>`) so the card's element
 * is the host itself and the chat DOM stays flat.
 */
@Component({
  selector: '[appCodeCard]',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './code-card.html',
  styleUrl: './code-card.scss',
  host: { class: 'code-item' },
})
export class CodeCardComponent {
  private prefsSvc = inject(PreferencesService);

  readonly codeEvent = input.required<CodeEvent>();
  /** The user asked to copy the source; the page owns the clipboard toast. */
  readonly copySource = output<string>();

  isSlow(): boolean {
    const ms = this.codeEvent().result?.duration_ms ?? 0;
    return ms >= this.prefsSvc.prefs().slowCodeThresholdMs;
  }
}
