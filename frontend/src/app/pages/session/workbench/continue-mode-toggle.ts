import { Component, input, output } from '@angular/core';
import { ContinueMode } from '../../../core/models/session.model';

export const CONTINUE_MODE_TOOLTIP =
  'Step: pause after each stage closes. Run: continue through handoffs (4 per check-in).';

/** Step | Run segmented toggle for the session header, bound to `session.continue_mode`. */
@Component({
  selector: 'app-continue-mode-toggle',
  standalone: true,
  template: `
    <div class="seg" role="group" aria-label="Continue mode" [title]="tooltip">
      <button type="button" [class.on]="mode() === 'step'" [attr.aria-pressed]="mode() === 'step'"
        [disabled]="disabled()" (click)="modeChange.emit('step')">Step</button>
      <button type="button" [class.on]="mode() === 'run'" [attr.aria-pressed]="mode() === 'run'"
        [disabled]="disabled()" (click)="modeChange.emit('run')">Run</button>
    </div>
  `,
  styles: `
    :host { display: inline-flex; }
    .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    button {
      border: 0; border-radius: 0; padding: 0.25rem 0.7rem; min-height: 30px;
      font: inherit; font-size: 0.82rem; background: var(--bg); color: var(--text-muted); cursor: pointer;
    }
    button + button { border-left: 1px solid var(--border); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button:hover:not(.on):not(:disabled) { background: var(--surface); color: var(--text); }
    button.on { background: var(--primary); color: var(--text-on-dark); font-weight: 600; }
    button:focus-visible { outline: 2px solid var(--primary-dark); outline-offset: -2px; }
  `,
})
export class ContinueModeToggleComponent {
  readonly mode = input.required<ContinueMode>();
  /** A change is in flight, or the session cannot take one. */
  readonly disabled = input(false);
  readonly tooltip = CONTINUE_MODE_TOOLTIP;
  readonly modeChange = output<ContinueMode>();
}
