/**
 * Pure column layout for the workbench step row: which grid step column
 * (1-based, after the label rail) each stage sits in, and, in a branch, the
 * parent's row above it. A stage is a chain of blocks that redo the same
 * step (`stages.ts`): a session without reruns or retries has one block per
 * column. Built on `laneLayout`, so a branch lines up under its parent the
 * same way the Paths row lines branches up under a session. No Angular, no
 * I/O.
 */
import { Block } from '../../../core/models/block.model';
import { laneLayout } from './branching';
import { RerunLinks, Stage, stages } from './stages';

/** A stage placed in a step column (1-based; the grid column is column + 1). */
export interface FlowCell extends Stage {
  /** The stage's latest block (the same as `latest`; kept as the cell's identity). */
  block: Block;
  column: number;
}

/** A parent stage in a branch's top row. */
export interface ParentCell extends FlowCell {
  /** At or after the branch point: what the branch replaces. */
  replaced: boolean;
}

export type BranchFlow =
  | { ok: true; parent: ParentCell[]; own: FlowCell[]; forkColumn: number; parentBlockOf: Map<string, string> }
  | { ok: false; error: string };

/** Stages placed in consecutive columns from `first`. */
function placed(list: Stage[], first: number): FlowCell[] {
  return list.map((stage, i) => ({ ...stage, block: stage.latest, column: first + i }));
}

/** A session's own row: every stage in first-block order, one column each. */
export function sessionCells(blocks: Block[], links: RerunLinks): FlowCell[] {
  return placed(stages(blocks, links), 1);
}

/** Block id → the step column of its stage. */
export function columnByBlock(cells: FlowCell[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const cell of cells) for (const v of cell.versions) out.set(v.block_id, cell.column);
  return out;
}

/**
 * A branch's two rows: all of the parent's stages (those from the branch
 * point on marked replaced), and the branch's OWN stages starting in the
 * branch point's column. Inherited blocks are not placed (their parent block
 * is in the top row); `parentBlockOf` maps each to the parent block it copies.
 * Each row is grouped with its own session's rerun links.
 */
export function branchFlow(
  parentBlocks: Block[], parentLinks: RerunLinks,
  branchBlocks: Block[], branchLinks: RerunLinks,
  parentId: string, forkedFromBlockId: string,
): BranchFlow {
  const parentCells = sessionCells(parentBlocks, parentLinks);
  const columns = columnByBlock(parentCells);
  const layout = laneLayout(columns, branchBlocks, branchLinks, parentId, forkedFromBlockId);
  if (!layout.ok) return { ok: false, error: layout.error };
  const forkColumn = columns.get(forkedFromBlockId)!;
  const parent = parentCells.map(cell => ({ ...cell, replaced: cell.column >= forkColumn }));
  const own: FlowCell[] = [];
  const parentBlockOf = new Map<string, string>();
  for (const cell of layout.cells) {
    if (cell.inherited) {
      for (const v of cell.versions) parentBlockOf.set(v.block_id, v.inherited_from!.block_id);
    } else {
      own.push({ ...cell });
    }
  }
  return { ok: true, parent, own, forkColumn, parentBlockOf };
}

/**
 * The branch row when the parent's flow could not be loaded: the branch's
 * own stages at their own positions (inherited ones are left as gaps, since
 * the parent row that would show them is a visible error placeholder).
 */
export function ownCellsWithoutParent(blocks: Block[], links: RerunLinks): FlowCell[] {
  // Inherited copies are staged apart from the branch's own blocks (an own
  // block that reruns an inherited item still gets its own column, as in
  // `laneLayout`); they come first in index order, so the own stages keep
  // the positions they have under the parent row.
  const inherited = blocks.filter(b => b.inherited_from != null);
  const own = blocks.filter(b => b.inherited_from == null);
  return placed([...stages(inherited, links), ...stages(own, links)], 1).filter(c => c.block.inherited_from == null);
}

/** The last step column a row reaches (0 when empty). */
export function lastColumn(cells: FlowCell[]): number {
  return cells.reduce((max, c) => Math.max(max, c.column), 0);
}
