/**
 * Pure helpers that group a session's blocks into stage columns. A rerun of a
 * stage opens a new work item linked to the original through `reruns`, and
 * each retry of an item is a new attempt block; all of these redo the same
 * stage, so the step row shows them in ONE column. No Angular, no I/O.
 *
 * - A block's chain root is its work item's root: follow `reruns` links to
 *   the first item. All attempts of all items in a chain share the column.
 * - An implicit block (no work item) is its own column.
 * - Columns are ordered by the index of the first block in the chain.
 * - Within a column the versions are in index order; the latest is the main
 *   card unless the user is looking at an earlier one.
 */
import { Block } from '../../../core/models/block.model';
import { WorkItemSummary } from '../../../core/models/session.model';
import { STATUS_GLYPH } from './block-matching';

/** Work item id → the id of the item it reruns (null for an original). */
export type RerunLinks = ReadonlyMap<number, number | null>;

/** One stage column: a chain of blocks that redo the same stage. */
export interface Stage {
  /** `item:<root item id>` for a work item chain, `block:<id>` for an implicit block. */
  key: string;
  /** The chain's first work item, or null for an implicit block. */
  rootItemId: number | null;
  /** Every block of the chain, index order (oldest first). */
  versions: Block[];
  /** The newest version (the last of `versions`). */
  latest: Block;
}

/** The rerun links of a work item list (an item without the field counts as an original). */
export function rerunLinks(items: Pick<WorkItemSummary, 'id' | 'reruns'>[]): Map<number, number | null> {
  return new Map(items.map(item => [item.id, item.reruns ?? null]));
}

/**
 * The first item of `itemId`'s rerun chain. An item the links don't list
 * ends the walk: nothing is known about it, so it is its own root (the
 * summary list is loaded with the session; an unlisted item is one whose
 * `work_item_changed` has not arrived yet). A cycle is a data error.
 */
export function chainRoot(itemId: number, links: RerunLinks): number {
  const seen = new Set<number>();
  let current = itemId;
  for (;;) {
    if (seen.has(current)) {
      throw new Error(`Work item #${itemId} has a rerun cycle through #${current}.`);
    }
    seen.add(current);
    const previous = links.get(current);
    if (previous == null) return current;
    current = previous;
  }
}

/** The column key of a block: its chain root, or the block itself when implicit. */
export function stageKey(block: Block, links: RerunLinks): string {
  return block.work_item_id === null
    ? `block:${block.block_id}`
    : `item:${chainRoot(block.work_item_id, links)}`;
}

/**
 * Group blocks into stages. The input order does not matter: blocks are
 * sorted by index, so versions are oldest first and stages are ordered by
 * their first block.
 */
export function stages(blocks: Block[], links: RerunLinks): Stage[] {
  const sorted = [...blocks].sort((a, b) => a.index - b.index);
  const byKey = new Map<string, Stage>();
  for (const block of sorted) {
    const key = stageKey(block, links);
    const stage = byKey.get(key);
    if (stage) {
      stage.versions.push(block);
      stage.latest = block;
    } else {
      byKey.set(key, {
        key,
        rootItemId: block.work_item_id === null ? null : chainRoot(block.work_item_id, links),
        versions: [block],
        latest: block,
      });
    }
  }
  return [...byKey.values()];
}

/** "v3": the block's 1-based position among the stage's versions. */
export function versionLabel(stage: Stage, block: Block): string {
  const i = stage.versions.findIndex(v => v.block_id === block.block_id);
  if (i < 0) throw new Error(`Block ${block.block_id} is not a version of stage ${stage.key}.`);
  return `v${i + 1}`;
}

/** "v2 ⚠": the strip entry for a version (status as a glyph, never colour alone). */
export function versionText(stage: Stage, block: Block): string {
  return `${versionLabel(stage, block)} ${STATUS_GLYPH[block.status]}`;
}

/**
 * The version a stage's card shows: the one the user is looking at
 * (`focusId`, a selected block or gate) when it is in this stage, else the
 * latest. So `#block:<id>` works for any version.
 */
export function shownVersion(stage: Stage, focusId: string | null): Block {
  if (focusId !== null) {
    const focused = stage.versions.find(v => v.block_id === focusId);
    if (focused) return focused;
  }
  return stage.latest;
}

/** The stage whose versions include the block, or null. */
export function stageOf(list: Stage[], blockId: string): Stage | null {
  return list.find(s => s.versions.some(v => v.block_id === blockId)) ?? null;
}

/**
 * "rerun of blk-0002" for a block whose work item reruns another (naming
 * that item's newest block, or the item when it has no block here),
 * "attempt 2" for a retry of the same item, both joined with " · " when
 * both apply, and null for a first attempt at an original item.
 */
export function rerunLabel(block: Block, links: RerunLinks, blocks: Block[]): string | null {
  const parts: string[] = [];
  const itemId = block.work_item_id;
  const previous = itemId === null ? null : links.get(itemId) ?? null;
  if (previous !== null) {
    const rerunBlock = [...blocks]
      .filter(b => b.work_item_id === previous)
      .sort((a, b) => b.index - a.index)[0];
    parts.push(`rerun of ${rerunBlock ? rerunBlock.block_id : `work item #${previous}`}`);
  }
  if (block.attempt > 1) parts.push(`attempt ${block.attempt}`);
  return parts.length ? parts.join(' · ') : null;
}
