import { Component, computed, input } from '@angular/core';
import { IconComponent } from '../../../shared/components/icon/icon';
import { ChatItem } from '../../../core/state/session-state.model';

export type Delegation = NonNullable<ChatItem['delegation']>;

/** The `agent_switch` command the runner sends when a work item goes back to its owner. */
export const RETURNED_FOR_CHANGES = 'returned_for_changes';

/**
 * One `agent_switch` in the chat: a delegation ("from → to `command`"), or
 * the runner returning to a work item's owner after a "Request changes"
 * ("returned to <owner> for changes", with the runner's reason).
 */
@Component({
  selector: 'app-delegation-card',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './delegation-card.html',
  styleUrl: './delegation-card.scss',
  host: { '[class.returned]': 'returned()' },
})
export class DelegationCardComponent {
  readonly delegation = input.required<Delegation>();

  readonly returned = computed(() => this.delegation().command === RETURNED_FOR_CHANGES);
}
