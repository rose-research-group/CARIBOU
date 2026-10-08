import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { ConfigService } from '../../core/services/config.service';
import { PreferencesService } from '../../core/services/preferences.service';
import { OllamaModelsResponse } from '../../core/models/session.model';
import { IconComponent } from '../../shared/components/icon/icon';

interface ServerSettings {
  caribou_home: string;
  sessions_dir: string;
  uploads_dir: string;
  env_file: string;
  api_keys: Record<string, string>;
  ollama_host: string;
  ollama_model: string;
  /** Live sandbox containers allowed at once; null is no limit. */
  max_active_sessions: number | null;
}

const MISSING_LIMIT =
  'The server did not report max_active_sessions, so settings cannot be saved. Update the CARIBOU server.';

/** The field's text for the server's limit ('' = no limit; also '' when omitted, which save refuses). */
function limitText(s: ServerSettings): string {
  return s.max_active_sessions == null ? '' : String(s.max_active_sessions);
}

/** The field's text → the limit (null = no limit), or an error to show. */
function parseMaxActiveSessions(text: string): { ok: true; value: number | null } | { ok: false; error: string } {
  const trimmed = text.trim();
  if (trimmed === '') return { ok: true, value: null };
  if (!/^\d+$/.test(trimmed) || Number(trimmed) < 1) {
    return { ok: false, error: 'Max active sessions must be a whole number of at least 1, or blank for no limit.' };
  }
  return { ok: true, value: Number(trimmed) };
}

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent],
  templateUrl: './settings.html',
  styleUrl: './settings.scss',
})
export class SettingsComponent implements OnInit {
  private http = inject(HttpClient);
  private router = inject(Router);
  configSvc = inject(ConfigService);
  prefsSvc = inject(PreferencesService);

  settings = signal<ServerSettings | null>(null);
  ollamaModels = signal<OllamaModelsResponse | null>(null);
  loading = signal(true);
  loadingOllama = signal(false);
  saving = signal(false);
  saveResult = signal<{ ok: boolean; message: string } | null>(null);

  // Editable fields
  sessionsDir = signal('');
  openaiKey = signal('');
  anthropicKey = signal('');
  deepseekKey = signal('');
  openrouterKey = signal('');
  ollamaHost = signal('');
  ollamaModel = signal('');
  /** Text of the "Max active sessions" field; blank means no limit. */
  maxActiveSessions = signal('');
  readonly MISSING_LIMIT = MISSING_LIMIT;

  showKeys: Record<string, boolean> = {
    openai: false,
    anthropic: false,
    deepseek: false,
    openrouter: false,
  };

  ngOnInit(): void {
    this.http.get<ServerSettings>('api/settings').subscribe({
      next: (s) => {
        this.settings.set(s);
        this.sessionsDir.set(s.sessions_dir);
        this.ollamaHost.set(s.ollama_host);
        this.ollamaModel.set(s.ollama_model);
        this.maxActiveSessions.set(limitText(s));
        this.loading.set(false);
        this.refreshOllamaModels();
      },
      error: () => this.loading.set(false),
    });
  }

  save(): void {
    this.saving.set(true);
    this.saveResult.set(null);
    const body: Record<string, string | number | null> = {};
    if (this.sessionsDir() !== this.settings()?.sessions_dir) {
      body['sessions_dir'] = this.sessionsDir();
    }
    if (this.openaiKey()) body['openai_api_key'] = this.openaiKey();
    if (this.anthropicKey()) body['anthropic_api_key'] = this.anthropicKey();
    if (this.deepseekKey()) body['deepseek_api_key'] = this.deepseekKey();
    if (this.openrouterKey()) body['openrouter_api_key'] = this.openrouterKey();
    if (this.ollamaHost() !== this.settings()?.ollama_host) {
      body['ollama_host'] = this.ollamaHost();
    }
    if (this.ollamaModel() !== this.settings()?.ollama_model) {
      body['ollama_model'] = this.ollamaModel();
    }

    const current = this.settings()?.max_active_sessions;
    if (current === undefined) {
      this.saving.set(false);
      this.saveResult.set({ ok: false, message: MISSING_LIMIT });
      return;
    }
    const limit = parseMaxActiveSessions(this.maxActiveSessions());
    if (!limit.ok) {
      this.saving.set(false);
      this.saveResult.set({ ok: false, message: limit.error });
      return;
    }
    if (limit.value !== current) {
      // null removes the limit on the server.
      body['max_active_sessions'] = limit.value;
    }

    if (Object.keys(body).length === 0) {
      this.saving.set(false);
      this.saveResult.set({ ok: true, message: 'Nothing to save.' });
      return;
    }

    this.http.patch<{ updated: string[] }>('api/settings', body).subscribe({
      next: (r) => {
        this.saving.set(false);
        this.openaiKey.set('');
        this.anthropicKey.set('');
        this.deepseekKey.set('');
        this.openrouterKey.set('');
        this.saveResult.set({ ok: true, message: `Saved: ${r.updated.join(', ')}` });
        // Reload settings to show updated masks
        this.http.get<ServerSettings>('api/settings').subscribe((s) => {
          this.settings.set(s);
          this.ollamaHost.set(s.ollama_host);
          this.ollamaModel.set(s.ollama_model);
          this.maxActiveSessions.set(limitText(s));
          this.refreshOllamaModels();
        });
      },
      error: (err) => {
        this.saving.set(false);
        this.saveResult.set({ ok: false, message: err?.error?.detail ?? 'Save failed.' });
      },
    });
  }

  goBack(): void {
    this.router.navigate(['/']);
  }

  refreshOllamaModels(): void {
    this.loadingOllama.set(true);
    this.configSvc.getOllamaModels().subscribe({
      next: (models) => {
        this.loadingOllama.set(false);
        this.ollamaModels.set(models);
        if (models.models.length && !models.models.includes(this.ollamaModel())) {
          this.ollamaModel.set(models.default_model);
        }
      },
      error: () => {
        this.loadingOllama.set(false);
        this.ollamaModels.set(null);
      },
    });
  }
}
