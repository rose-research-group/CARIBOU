import { Injectable } from '@angular/core';

// v2: chat items/artifacts carry seq- and path-based identity; v1 entries
// predate that and are never read.
const PREFIX = 'caribou:session-cache:v2:';
const MAX_ITEMS = 400;

export interface CachedSessionState {
  chatItems: unknown[];
  artifacts: unknown[];
  errorLog: unknown[];
  statusLog: unknown[];
  /** Highest event seq whose effects are reflected in this snapshot. */
  lastSeq: number;
  cachedAt: number;
}

/**
 * sessionStorage snapshot of a session page for refresh recovery.
 * Storage errors (quota, disabled storage) and corrupt entries propagate.
 */
@Injectable({ providedIn: 'root' })
export class SessionCacheService {
  read(sessionId: string): CachedSessionState | null {
    const raw = sessionStorage.getItem(PREFIX + sessionId);
    if (raw === null) return null;
    const parsed = JSON.parse(raw) as CachedSessionState;
    if (!parsed || typeof parsed !== 'object' || typeof parsed.lastSeq !== 'number') {
      throw new Error(`Corrupt session cache entry for session ${sessionId}`);
    }
    return parsed;
  }

  write(sessionId: string, state: Omit<CachedSessionState, 'cachedAt'>): void {
    const payload: CachedSessionState = {
      chatItems: (state.chatItems ?? []).slice(-MAX_ITEMS),
      artifacts: state.artifacts ?? [],
      errorLog: (state.errorLog ?? []).slice(-50),
      statusLog: (state.statusLog ?? []).slice(-50),
      lastSeq: state.lastSeq,
      cachedAt: Date.now(),
    };
    sessionStorage.setItem(PREFIX + sessionId, JSON.stringify(payload));
  }

  clear(sessionId: string): void {
    sessionStorage.removeItem(PREFIX + sessionId);
  }
}
