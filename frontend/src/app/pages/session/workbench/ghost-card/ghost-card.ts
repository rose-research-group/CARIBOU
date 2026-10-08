import { Component, DestroyRef, computed, inject, input, signal } from '@angular/core';
import { formatElapsed } from '../activity';

/**
 * "The agent is working" indicator: the agent, a live activity line, the
 * time since the work started and the queued-message count. `variant`
 * 'ghost' is the dashed card at the end of the step row; 'inline' is the
 * compact strip on the block the in-flight message is about.
 */
@Component({
  selector: 'app-ghost-card',
  standalone: true,
  templateUrl: './ghost-card.html',
  styleUrl: './ghost-card.scss',
  host: { '[class.inline]': "variant() === 'inline'" },
})
export class GhostCardComponent {
  readonly agent = input.required<string>();
  readonly activity = input.required<string>();
  /** When the agent started working (ms); null when not known. */
  readonly since = input.required<number | null>();
  readonly queued = input(0);
  readonly variant = input<'ghost' | 'inline'>('ghost');

  private readonly now = signal(Date.now());
  readonly elapsed = computed(() => {
    const since = this.since();
    return since === null ? null : formatElapsed(this.now() - since);
  });

  constructor() {
    const timer = setInterval(() => this.now.set(Date.now()), 1000);
    inject(DestroyRef).onDestroy(() => clearInterval(timer));
  }
}
