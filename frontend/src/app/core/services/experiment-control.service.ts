import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  ArtifactsData,
  CancelData,
  CheckpointData,
  CheckpointsData,
  EventsData,
  MachineResponse,
  PresetResolveData,
  PresetResolveRequest,
  PresetSummary,
  ResumeData,
  StatusData,
  SubmitData,
  VerifyData,
} from '../models/experiment-control.model';

const CONTROL_TOKEN_STORAGE_KEY = 'caribou:control:bearer-token:v1';

@Injectable({ providedIn: 'root' })
export class ExperimentControlService {
  private http = inject(HttpClient);
  private accessToken = signal(this.loadAccessToken());

  readonly hasAccessToken = computed(() => this.accessToken().length > 0);

  setAccessToken(value: string): void {
    const token = value.trim();
    this.accessToken.set(token);
    if (token) sessionStorage.setItem(CONTROL_TOKEN_STORAGE_KEY, token);
    else sessionStorage.removeItem(CONTROL_TOKEN_STORAGE_KEY);
  }

  clearAccessToken(): void {
    this.setAccessToken('');
  }

  schema(): Observable<MachineResponse<{ name: string; schema: Record<string, unknown> }>> {
    return this.http.get<MachineResponse<{ name: string; schema: Record<string, unknown> }>>(
      'api/control/schema/experiment',
      { headers: this.controlHeaders() },
    );
  }

  presets(): Observable<MachineResponse<{ presets: PresetSummary[] }>> {
    return this.http.get<MachineResponse<{ presets: PresetSummary[] }>>('api/control/presets', {
      headers: this.controlHeaders(),
    });
  }

  resolvePreset(
    presetId: string,
    request: PresetResolveRequest,
  ): Observable<MachineResponse<PresetResolveData>> {
    return this.http.post<MachineResponse<PresetResolveData>>(
      `api/control/presets/${encodeURIComponent(presetId)}/resolve`,
      request,
      { headers: this.controlHeaders() },
    );
  }

  validate(specification: unknown): Observable<MachineResponse<Record<string, unknown>>> {
    return this.http.post<MachineResponse<Record<string, unknown>>>(
      'api/control/experiments/validate',
      specification,
      { headers: this.controlHeaders() },
    );
  }

  plan(specification: unknown): Observable<MachineResponse<Record<string, unknown>>> {
    return this.http.post<MachineResponse<Record<string, unknown>>>(
      'api/control/experiments/plan',
      specification,
      { headers: this.controlHeaders() },
    );
  }

  submit(
    specification: unknown,
    idempotencyKey: string,
    expectedPlanHash?: string,
  ): Observable<MachineResponse<SubmitData>> {
    return this.http.post<MachineResponse<SubmitData>>(
      'api/control/experiments',
      {
        specification,
        idempotency_key: idempotencyKey,
        expected_plan_hash: expectedPlanHash || null,
      },
      { headers: this.controlHeaders() },
    );
  }

  cloneExperiment(
    experimentId: string,
    evaluatorProvider: string,
    evaluatorModel: string,
    reason?: string,
  ): Observable<MachineResponse<Record<string, unknown>>> {
    return this.http.post<MachineResponse<Record<string, unknown>>>(
      `api/control/experiments/${encodeURIComponent(experimentId)}/clone`,
      {
        evaluator_provider: evaluatorProvider,
        evaluator_model: evaluatorModel,
        reason: reason?.trim() || null,
      },
      { headers: this.controlHeaders() },
    );
  }

  status(runId: string): Observable<MachineResponse<StatusData>> {
    return this.http.get<MachineResponse<StatusData>>(`api/control/runs/${runId}`, {
      headers: this.controlHeaders(),
    });
  }

  events(runId: string, after: number): Observable<MachineResponse<EventsData>> {
    return this.http.get<MachineResponse<EventsData>>(`api/control/runs/${runId}/events`, {
      headers: this.controlHeaders(),
      params: { after: String(after), limit: '1000' },
    });
  }

  cancel(runId: string, reason: string): Observable<MachineResponse<CancelData>> {
    return this.http.post<MachineResponse<CancelData>>(
      `api/control/runs/${runId}/cancel`,
      { reason },
      { headers: this.controlHeaders() },
    );
  }

  requestCheckpoint(
    runId: string,
    idempotencyKey: string,
    reason: string,
  ): Observable<MachineResponse<CheckpointData>> {
    return this.http.post<MachineResponse<CheckpointData>>(
      `api/control/runs/${runId}/checkpoint`,
      { idempotency_key: idempotencyKey, reason },
      { headers: this.controlHeaders() },
    );
  }

  checkpoints(runId: string): Observable<MachineResponse<CheckpointsData>> {
    return this.http.get<MachineResponse<CheckpointsData>>(
      `api/control/runs/${runId}/checkpoints`,
      { headers: this.controlHeaders() },
    );
  }

  resume(
    runId: string,
    checkpointId: string,
    idempotencyKey: string,
  ): Observable<MachineResponse<ResumeData>> {
    return this.http.post<MachineResponse<ResumeData>>(
      `api/control/runs/${runId}/resume`,
      { checkpoint_id: checkpointId, idempotency_key: idempotencyKey },
      { headers: this.controlHeaders() },
    );
  }

  artifacts(runId: string): Observable<MachineResponse<ArtifactsData>> {
    return this.http.get<MachineResponse<ArtifactsData>>(`api/control/runs/${runId}/artifacts`, {
      headers: this.controlHeaders(),
    });
  }

  verifyArtifacts(runId: string): Observable<MachineResponse<VerifyData>> {
    return this.http.post<MachineResponse<VerifyData>>(
      `api/control/runs/${runId}/artifacts/verify`,
      {},
      { headers: this.controlHeaders() },
    );
  }

  downloadArtifact(runId: string, artifactId: string): Observable<Blob> {
    return this.http.get(`api/control/runs/${runId}/artifacts/${artifactId}/download`, {
      headers: this.controlHeaders(),
      responseType: 'blob',
    });
  }

  private controlHeaders(): HttpHeaders {
    return new HttpHeaders({ 'X-Caribou-Control-Token': this.accessToken() });
  }

  private loadAccessToken(): string {
    try {
      return sessionStorage.getItem(CONTROL_TOKEN_STORAGE_KEY) ?? '';
    } catch {
      return '';
    }
  }
}
