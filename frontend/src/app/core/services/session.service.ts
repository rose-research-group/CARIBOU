import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import {
  Artifact, CodeEvent, EvaluationResult, Message, MemoryState, Session, SessionCreateRequest,
  SessionForkRequest, SessionResumeRequest,
  EvaluatorModelState, EvaluatorModelUpdateRequest,
  WorkItemDetail, WorkItemReviewResult, WorkItemSummary, SessionBriefFields,
  WorkItemCreateRequest, HumanReviewRequest, BranchRequest, BranchSummary,
} from '../models/session.model';
import { BlocksResponse } from '../models/block.model';

@Injectable({ providedIn: 'root' })
export class SessionService {
  private http = inject(HttpClient);

  readonly sessions = signal<Session[]>([]);
  readonly currentSession = signal<Session | null>(null);

  loadSessions(): Observable<Session[]> {
    return this.http.get<Session[]>('api/sessions').pipe(
      tap(s => this.sessions.set(s))
    );
  }

  createSession(config: SessionCreateRequest): Observable<Session> {
    return this.http.post<Session>('api/sessions', config).pipe(
      tap(s => {
        this.sessions.update(all => [s, ...all]);
        this.currentSession.set(s);
      })
    );
  }

  resumeSession(id: string, request: SessionResumeRequest): Observable<Session> {
    return this.http.post<Session>(`api/sessions/${id}/resume`, request).pipe(
      tap(s => this.updateLocal(s))
    );
  }

  forkSession(id: string, request: SessionForkRequest): Observable<Session> {
    return this.http.post<Session>(`api/sessions/${id}/fork`, request).pipe(
      tap(s => this.sessions.update(all => [s, ...all]))
    );
  }

  retryRecovery(id: string, request: SessionResumeRequest): Observable<Session> {
    return this.http.post<Session>(`api/sessions/${id}/recovery/retry`, request).pipe(
      tap(s => this.updateLocal(s))
    );
  }

  acceptPartialRecovery(id: string): Observable<Session> {
    return this.http.post<Session>(`api/sessions/${id}/recovery/accept-partial`, {}).pipe(
      tap(s => this.updateLocal(s))
    );
  }

  getSession(id: string): Observable<Session> {
    return this.http.get<Session>(`api/sessions/${id}`).pipe(
      tap(s => {
        this.currentSession.set(s);
        this.sessions.update(all => all.map(x => x.id === s.id ? s : x));
      })
    );
  }

  deleteSession(id: string): Observable<void> {
    return this.http.delete<void>(`api/sessions/${id}`).pipe(
      tap(() => {
        this.sessions.update(all => all.filter(s => s.id !== id));
        if (this.currentSession()?.id === id) {
          this.currentSession.set(null);
        }
      })
    );
  }

  getMessages(id: string, offset = 0, limit = 500): Observable<Message[]> {
    return this.http.get<Message[]>(`api/sessions/${id}/messages`, {
      params: { offset: String(offset), limit: String(limit) }
    });
  }

  getArtifacts(id: string): Observable<Artifact[]> {
    return this.http.get<Artifact[]>(`api/sessions/${id}/artifacts`);
  }

  getCodeEvents(id: string): Observable<CodeEvent[]> {
    return this.http.get<CodeEvent[]>(`api/sessions/${id}/code_events`);
  }

  getWorkItems(id: string): Observable<WorkItemSummary[]> {
    return this.http.get<WorkItemSummary[]>(`api/sessions/${id}/work-items`);
  }

  getWorkItem(id: string, itemId: number): Observable<WorkItemDetail> {
    return this.http.get<WorkItemDetail>(`api/sessions/${id}/work-items/${itemId}`);
  }

  reviewWorkItem(id: string, itemId: number): Observable<WorkItemReviewResult> {
    return this.http.post<WorkItemReviewResult>(
      `api/sessions/${id}/work-items/${itemId}/review`,
      {},
    );
  }

  /** Open a ticket as the user. Its `work_item_changed` event updates the store. */
  createWorkItem(id: string, request: WorkItemCreateRequest): Observable<WorkItemDetail> {
    return this.http.post<WorkItemDetail>(`api/sessions/${id}/work-items`, request);
  }

  /** Record the user's approve/reject. Its `work_item_changed` event updates the store. */
  humanReviewWorkItem(id: string, itemId: number, request: HumanReviewRequest): Observable<WorkItemDetail> {
    return this.http.post<WorkItemDetail>(
      `api/sessions/${id}/work-items/${itemId}/human-review`,
      request,
    );
  }

  getBlocks(id: string): Observable<BlocksResponse> {
    return this.http.get<BlocksResponse>(`api/sessions/${id}/blocks`);
  }

  /**
   * Another session's record, without touching `currentSession` (the parent
   * of a branch, for its name). `getSession` is for the page's own session.
   */
  fetchSession(id: string): Observable<Session> {
    return this.http.get<Session>(`api/sessions/${id}`);
  }

  /** Branch a new session from a block's entry checkpoint. 201 with the child. */
  branchFromBlock(id: string, blockId: string, request: BranchRequest): Observable<Session> {
    return this.http.post<Session>(
      `api/sessions/${id}/blocks/${encodeURIComponent(blockId)}/branch`,
      request,
    ).pipe(
      tap(s => this.sessions.update(all => [s, ...all]))
    );
  }

  /** The session's direct child branches, sorted by created_at. */
  getBranches(id: string): Observable<BranchSummary[]> {
    return this.http.get<BranchSummary[]>(`api/sessions/${id}/branches`);
  }

  submitBriefDecision(
    id: string,
    decision: 'accept' | 'reject' | 'edit',
    options?: { reason?: string; brief?: Partial<SessionBriefFields> },
  ): Observable<void> {
    return this.http.post<void>(`api/sessions/${id}/brief/decision`, {
      decision,
      reason: options?.reason ?? null,
      brief: options?.brief ?? null,
    });
  }

  artifactDownloadUrl(sessionId: string, artifactId: string): string {
    const base = document.baseURI.replace(/\/$/, '');
    return `${base}/api/sessions/${sessionId}/artifacts/${artifactId}/download`;
  }

  notebookDownloadUrl(sessionId: string): string {
    const base = document.baseURI.replace(/\/$/, '');
    return `${base}/api/sessions/${sessionId}/notebook`;
  }

  getMemoryState(id: string): Observable<MemoryState> {
    return this.http.get<MemoryState>(`api/sessions/${id}/memory`);
  }

  evaluate(id: string): Observable<EvaluationResult> {
    return this.http.post<EvaluationResult>(`api/sessions/${id}/evaluate`, {});
  }

  getEvaluatorModel(id: string): Observable<EvaluatorModelState> {
    return this.http.get<EvaluatorModelState>(`api/sessions/${id}/evaluator-model`);
  }

  updateEvaluatorModel(
    id: string,
    request: EvaluatorModelUpdateRequest,
  ): Observable<EvaluatorModelState> {
    return this.http.patch<EvaluatorModelState>(
      `api/sessions/${id}/evaluator-model`,
      request,
    ).pipe(
      tap(state => {
        this.sessions.update(all => all.map(s =>
          s.id === id ? { ...s, evaluator_model: state } : s
        ));
        if (this.currentSession()?.id === id) {
          this.currentSession.update(s => s ? { ...s, evaluator_model: state } : s);
        }
      })
    );
  }

  updateLocal(session: Session): void {
    this.currentSession.set(session);
    this.sessions.update(all => all.map(s => s.id === session.id ? session : s));
  }

  clearCurrentSession(): void {
    this.currentSession.set(null);
  }
}
