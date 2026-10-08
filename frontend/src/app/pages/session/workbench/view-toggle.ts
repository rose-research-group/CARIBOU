import { Component, input, output } from '@angular/core';
import { SPLIT_MIN_VIEWPORT_PX } from '../split-pane/split-ratio';

export type SessionView = 'chat' | 'split' | 'workbench';

/** Chat | Split | Workbench segmented toggle for the session header. */
@Component({
  selector: 'app-view-toggle',
  standalone: true,
  template: `
    <div class="seg" role="group" aria-label="Session view">
      <button type="button" [class.on]="view() === 'chat'" [attr.aria-pressed]="view() === 'chat'"
        (click)="viewChange.emit('chat')">Chat</button>
      <button type="button" [class.on]="view() === 'split'" [attr.aria-pressed]="view() === 'split'"
        [attr.aria-disabled]="!splitAvailable()"
        [title]="splitAvailable() ? 'Chat and workbench side by side' : splitUnavailableReason"
        (click)="splitAvailable() && viewChange.emit('split')">Split</button>
      <button type="button" [class.on]="view() === 'workbench'" [attr.aria-pressed]="view() === 'workbench'"
        (click)="viewChange.emit('workbench')">Workbench</button>
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
    /* aria-disabled (not disabled) so the reason tooltip still shows. */
    button[aria-disabled="true"] { opacity: 0.5; cursor: not-allowed; }
    button:hover:not(.on):not([aria-disabled="true"]) { background: var(--surface); color: var(--text); }
    button.on { background: var(--primary); color: var(--text-on-dark); font-weight: 600; }
    button:focus-visible { outline: 2px solid var(--primary-dark); outline-offset: -2px; }
  `,
})
export class ViewToggleComponent {
  readonly view = input.required<SessionView>();
  /** False below the split view's minimum window width. */
  readonly splitAvailable = input(true);
  readonly splitUnavailableReason = `Split view needs a window at least ${SPLIT_MIN_VIEWPORT_PX}px wide.`;
  readonly viewChange = output<SessionView>();
}
