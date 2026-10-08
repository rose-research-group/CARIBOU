import { Component, computed, input, output } from '@angular/core';
import { Block } from '../../../../core/models/block.model';
import { GATE_GLYPH, Gate, gateAriaLabel } from '../gates';

/**
 * The review gate after a block: a small circle whose shape and glyph (not
 * only its colour) give the state. The connector across the gap to the next
 * column is drawn here too: solid when open, dashed when closed (required QC,
 * not yet approved). Placed by the workbench in the block's grid cell.
 */
@Component({
  selector: 'app-gate-node',
  standalone: true,
  templateUrl: './gate-node.html',
  styleUrl: './gate-node.scss',
})
export class GateNodeComponent {
  readonly block = input.required<Block>();
  readonly gate = input.required<Gate>();
  readonly selected = input(false);
  /** Draw the connector on past the gate (a next step, or the ghost card). */
  readonly continues = input(false);
  /** The next column holds this attempt's retry: draw the thin retry line. */
  readonly retryNext = input(false);
  readonly inherited = input(false);
  readonly pick = output<void>();

  readonly glyph = computed(() => GATE_GLYPH[this.gate().state]);
  readonly label = computed(() => gateAriaLabel(this.block(), this.gate()));
}
