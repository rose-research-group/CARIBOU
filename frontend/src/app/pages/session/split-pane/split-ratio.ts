/** Pure helpers for the split view's divider. No Angular, no I/O. */

export const SPLIT_MIN = 0.3;
export const SPLIT_MAX = 0.7;
export const SPLIT_DEFAULT = 0.5;
export const SPLIT_STEP = 0.02;
export const SPLIT_STORAGE_KEY = 'caribou.session.splitRatio';
/** Below this viewport width the split view is unavailable. */
export const SPLIT_MIN_VIEWPORT_PX = 1100;

export function clampSplit(ratio: number): number {
  return Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, ratio));
}

/** A stored ratio, or null when absent or not a number in range. */
export function parseStoredSplit(raw: string | null): number | null {
  if (raw === null) return null;
  const n = Number(raw);
  return Number.isFinite(n) && n >= SPLIT_MIN && n <= SPLIT_MAX ? n : null;
}
