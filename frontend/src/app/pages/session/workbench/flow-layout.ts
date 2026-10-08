/**
 * Pure column layout for the workbench step row: which grid step column
 * (1-based, after the label rail) each block sits in, and, in a branch, the
 * parent's row above it. Built on `laneLayout`, so a branch lines up under
 * its parent the same way the Paths row lines branches up under a session.
 * No Angular, no I/O.
 */
import { Block } from '../../../core/models/block.model';
import { laneLayout } from './branching';

/** A block placed in a step column (1-based; the grid column is column + 1). */
export interface FlowCell {
  block: Block;
  column: number;
}

/** A parent block in a branch's top row. */
export interface ParentCell extends FlowCell {
  /** At or after the branch point: what the branch replaces. */
  replaced: boolean;
}

export type BranchFlow =
  | { ok: true; parent: ParentCell[]; own: FlowCell[]; forkColumn: number; parentBlockOf: Map<string, string> }
  | { ok: false; error: string };

/** A session's own row: every block in index order, one column each. */
export function sessionCells(blocks: Block[]): FlowCell[] {
  return blocks.map((block, i) => ({ block, column: i + 1 }));
}

/**
 * A branch's two rows: all of the parent's blocks (those from the branch
 * point on marked replaced), and the branch's OWN blocks starting in the
 * branch point's column. Inherited blocks are not placed (their parent block
 * is in the top row); `parentBlockOf` maps each to the parent block it copies.
 */
export function branchFlow(
  parentBlocks: Block[], branchBlocks: Block[], parentId: string, forkedFromBlockId: string,
): BranchFlow {
  const parentSorted = [...parentBlocks].sort((a, b) => a.index - b.index);
  const layout = laneLayout(parentSorted, branchBlocks, parentId, forkedFromBlockId);
  if (!layout.ok) return { ok: false, error: layout.error };
  const forkColumn = parentSorted.findIndex(b => b.block_id === forkedFromBlockId) + 1;
  const parent = parentSorted.map((block, i) => ({ block, column: i + 1, replaced: i + 1 >= forkColumn }));
  const own: FlowCell[] = [];
  const parentBlockOf = new Map<string, string>();
  for (const cell of layout.cells) {
    if (cell.inherited) parentBlockOf.set(cell.block.block_id, cell.block.inherited_from!.block_id);
    else own.push({ block: cell.block, column: cell.column });
  }
  return { ok: true, parent, own, forkColumn, parentBlockOf };
}

/**
 * The branch row when the parent's flow could not be loaded: the branch's
 * own blocks at their own positions (inherited ones are left as gaps, since
 * the parent row that would show them is a visible error placeholder).
 */
export function ownCellsWithoutParent(blocks: Block[]): FlowCell[] {
  return sessionCells(blocks).filter(c => c.block.inherited_from == null);
}

/** The last step column a row reaches (0 when empty). */
export function lastColumn(cells: FlowCell[]): number {
  return cells.reduce((max, c) => Math.max(max, c.column), 0);
}
