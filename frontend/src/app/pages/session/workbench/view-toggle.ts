import { Component, input, output } from '@angular/core';

export type SessionView = 'chat' | 'workbench';

/** Chat | Workbench segmented toggle for the session header. */
@Component({
  selector: 'app-view-toggle',
  standalone: true,
  template: `
    <div class="seg" role="group" aria-label="Session view">
      <button type="button" [class.on]="view() === 'chat'" [attr.aria-pressed]="view() === 'chat'"
        (click)="viewChange.emit('chat')">Chat</button>
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
    button:hover:not(.on) { background: var(--surface); color: var(--text); }
    button.on { background: var(--primary); color: var(--text-on-dark); font-weight: 600; }
    button:focus-visible { outline: 2px solid var(--primary-dark); outline-offset: -2px; }
  `,
})
export class ViewToggleComponent {
  readonly view = input.required<SessionView>();
  readonly viewChange = output<SessionView>();
}
