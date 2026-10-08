import { Component, input, output } from '@angular/core';
import { QueuedMessage } from '../../../core/models/session.model';
import { IconComponent } from '../../../shared/components/icon/icon';

/**
 * The messages the server holds until the agent is ready, shown directly
 * above the chat input. It renders the store's queue as is: a ✕ asks the page
 * to remove an item, and the item goes when the server's next
 * `message_queue_changed` event drops it (never optimistically).
 */
@Component({
  selector: 'app-queue-bar',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './queue-bar.html',
  styleUrl: './queue-bar.scss',
})
export class QueueBarComponent {
  readonly queue = input.required<QueuedMessage[]>();
  /** Ids whose removal was sent and not yet confirmed. */
  readonly pendingRemovals = input.required<Set<string>>();
  readonly remove = output<string>();
}
