import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { ConfigService } from '../../core/services/config.service';
import { PreferencesService } from '../../core/services/preferences.service';
import { OllamaModelsResponse, ServerSettings } from '../../core/models/session.model';
import { IconComponent } from '../../shared/components/icon/icon';

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
  slurmPartition = signal('');

  showKeys: Record<string, boolean> = {
    openai: false,
    anthropic: false,
    deepseek: false,
    openrouter: false,
  };

  loadError = signal<string | null>(null);

  ngOnInit(): void {
    this.configSvc.getSettings().subscribe({
      next: (s) => {
        this.settings.set(s);
        this.sessionsDir.set(s.sessions_dir);
        this.ollamaHost.set(s.ollama_host);
        this.ollamaModel.set(s.ollama_model);
        this.slurmPartition.set(s.slurm_partition ?? '');
        this.loading.set(false);
        this.refreshOllamaModels();
      },
      error: (err) => {
        this.loading.set(false);
        this.loadError.set(err?.error?.detail ?? 'Failed to load server settings.');
      },
    });
  }

  save(): void {
    this.saving.set(true);
    this.saveResult.set(null);
    const body: Record<string, string> = {};
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
    if (this.slurmPartition() !== (this.settings()?.slurm_partition ?? '')) {
      body['slurm_partition'] = this.slurmPartition();
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
        this.configSvc.getSettings().subscribe({
          next: (s) => {
            this.settings.set(s);
            this.ollamaHost.set(s.ollama_host);
            this.ollamaModel.set(s.ollama_model);
            this.slurmPartition.set(s.slurm_partition ?? '');
            this.refreshOllamaModels();
          },
          error: (err) => {
            this.saveResult.set({
              ok: false,
              message: err?.error?.detail ?? 'Saved, but reloading settings failed.',
            });
          },
        });
      },
      error: (err) => {
        this.saving.set(false);
        this.saveResult.set({ ok: false, message: this.describeError(err, 'Save failed.') });
      },
    });
  }

  /** FastAPI returns a string detail for HTTPException and a list for body validation. */
  private describeError(err: unknown, fallbackMessage: string): string {
    const detail = (err as { error?: { detail?: unknown } } | null)?.error?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (d && typeof d === 'object' && 'msg' in d ? String(d.msg) : String(d)))
        .join('; ');
    }
    return fallbackMessage;
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
