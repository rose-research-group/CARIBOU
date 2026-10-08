import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { tap } from 'rxjs/operators';
import { Observable } from 'rxjs';
import {
  AgentBlueprint,
  LLMBackend,
  OllamaModelsResponse,
  OpenRouterCatalogue,
  OpenRouterEndpointsResponse,
  PythonEnvironmentCandidate,
  ServerSettings,
  ServerStatus,
} from '../models/session.model';

@Injectable({ providedIn: 'root' })
export class ConfigService {
  private http = inject(HttpClient);

  readonly serverStatus = signal<ServerStatus | null>(null);
  readonly blueprints = signal<AgentBlueprint[]>([]);
  readonly backends = signal<LLMBackend[]>([]);
  readonly ollamaModels = signal<OllamaModelsResponse | null>(null);
  readonly openRouterCatalogue = signal<OpenRouterCatalogue | null>(null);
  readonly pythonEnvironments = signal<PythonEnvironmentCandidate[]>([]);
  readonly settings = signal<ServerSettings | null>(null);

  loadAll(): Observable<unknown> {
    return new Observable((obs) => {
      let done = 0;
      const check = () => {
        if (++done === 5) {
          obs.next(null);
          obs.complete();
        }
      };
      this.http.get<ServerStatus>('api/status').subscribe((s) => {
        this.serverStatus.set(s);
        check();
      });
      this.http.get<AgentBlueprint[]>('api/config/blueprints').subscribe((b) => {
        this.blueprints.set(b);
        check();
      });
      this.http.get<LLMBackend[]>('api/config/backends').subscribe((b) => {
        this.backends.set(b);
        check();
      });
      this.http.get<OllamaModelsResponse>('api/config/ollama/models').subscribe({
        next: (o) => {
          this.ollamaModels.set(o);
          check();
        },
        error: () => {
          this.ollamaModels.set(null);
          check();
        },
      });
      this.getPythonEnvironments().subscribe({
        next: () => check(),
        error: () => {
          this.pythonEnvironments.set([]);
          check();
        },
      });
    });
  }

  getSettings(): Observable<ServerSettings> {
    return this.http
      .get<ServerSettings>('api/settings')
      .pipe(tap((s) => this.settings.set(s)));
  }

  getStatus(): Observable<ServerStatus> {
    return this.http.get<ServerStatus>('api/status').pipe(tap((s) => this.serverStatus.set(s)));
  }

  getBlueprints(): Observable<AgentBlueprint[]> {
    return this.http
      .get<AgentBlueprint[]>('api/config/blueprints')
      .pipe(tap((b) => this.blueprints.set(b)));
  }

  getBackends(): Observable<LLMBackend[]> {
    return this.http
      .get<LLMBackend[]>('api/config/backends')
      .pipe(tap((b) => this.backends.set(b)));
  }

  getOllamaModels(): Observable<OllamaModelsResponse> {
    return this.http
      .get<OllamaModelsResponse>('api/config/ollama/models')
      .pipe(tap((o) => this.ollamaModels.set(o)));
  }

  startOllama(): Observable<OllamaModelsResponse> {
    return this.http
      .post<OllamaModelsResponse>('api/config/ollama/start', {})
      .pipe(tap((o) => this.ollamaModels.set(o)));
  }

  getOpenRouterModels(refresh = false): Observable<OpenRouterCatalogue> {
    return this.http
      .get<OpenRouterCatalogue>('api/config/openrouter/models', {
        params: { refresh: String(refresh) },
      })
      .pipe(tap((catalogue) => this.openRouterCatalogue.set(catalogue)));
  }

  getOpenRouterEndpoints(modelId: string): Observable<OpenRouterEndpointsResponse> {
    return this.http.get<OpenRouterEndpointsResponse>('api/config/openrouter/endpoints', {
      params: { model_id: modelId },
    });
  }

  getPythonEnvironments(): Observable<PythonEnvironmentCandidate[]> {
    return this.http
      .get<PythonEnvironmentCandidate[]>('api/config/python-environments')
      .pipe(tap((items) => this.pythonEnvironments.set(items)));
  }

  validatePythonEnvironment(path: string): Observable<PythonEnvironmentCandidate> {
    return this.http.post<PythonEnvironmentCandidate>(
      'api/config/python-environments/validate',
      { path },
    );
  }
}
