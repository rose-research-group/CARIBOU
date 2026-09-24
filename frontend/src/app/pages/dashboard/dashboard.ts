import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { SessionService } from '../../core/services/session.service';
import { ConfigService } from '../../core/services/config.service';
import { DatasetService } from '../../core/services/dataset.service';
import { ToastService } from '../../core/services/toast.service';
import {
  Session, SessionCreateRequest, SessionForkRequest, Dataset, PythonEnvironmentCandidate
} from '../../core/models/session.model';
import { StatusIndicatorComponent } from '../../shared/components/status-indicator/status-indicator';
import { IconComponent } from '../../shared/components/icon/icon';
import { TooltipDirective } from '../../shared/directives/tooltip.directive';
import { navigateTabToSession, reserveNewTab } from '../../core/utils/app-navigation';

type DatasetSource = 'existing' | 'upload' | 'hpc';
type SortKey = 'newest' | 'oldest' | 'status' | 'turns';
type StatusFilter = 'all' | 'running' | 'idle' | 'stopped' | 'error' | 'initializing' | 'recovering';

const DEFAULT_AGENT_SYSTEM = 'caribou_fully_connected_v2';

export interface QuickStartPreset {
  id: string;
  title: string;
  subtitle: string;
  icon: string;
  values: Partial<SessionCreateRequest> & { initial_prompt: string };
}

const PRESETS: QuickStartPreset[] = [
  {
    id: 'scrna-explore',
    title: 'scRNA-seq exploration',
    subtitle: 'Interactive · full agent system',
    icon: 'microscope',
    values: {
      mode: 'interactive',
      run_mode: 'full_system',
      agent_system: DEFAULT_AGENT_SYSTEM,
      sandbox_type: 'singularity',
      initial_prompt:
        'Load the dataset and give me a first-pass QC summary: shape, obs/var keys, ' +
        'total counts distribution, and any obvious quality flags.',
    },
  },
  {
    id: 'cell-typing-auto',
    title: 'Cell typing (auto)',
    subtitle: 'Auto · 20 turns · full agent system',
    icon: 'dna',
    values: {
      mode: 'auto',
      run_mode: 'full_system',
      agent_system: DEFAULT_AGENT_SYSTEM,
      sandbox_type: 'singularity',
      max_turns: 20,
      initial_prompt:
        'Run standard cell type annotation on this dataset and produce a UMAP colored ' +
        'by predicted cell type. Save all outputs to /workspace/outputs.',
    },
  },
  {
    id: 'custom',
    title: 'Custom',
    subtitle: 'Configure every field yourself',
    icon: 'sliders',
    values: { initial_prompt: '' },
  },
];

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RouterLink,
    StatusIndicatorComponent,
    IconComponent,
    TooltipDirective,
  ],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class DashboardComponent implements OnInit {
  private router = inject(Router);
  sessionSvc = inject(SessionService);
  configSvc = inject(ConfigService);
  datasetSvc = inject(DatasetService);
  private toasts = inject(ToastService);

  datasets = signal<Dataset[]>([]);
  showCreateDialog = signal(false);
  creating = signal(false);
  createError = signal<string | null>(null);
  loadingOpenRouter = signal(false);
  openRouterError = signal<string | null>(null);
  pendingDeleteId = signal<string | null>(null);
  pendingBulkDelete = signal(false);
  datasetSource = signal<DatasetSource>('existing');
  hpcDatasetPath = signal('');
  hpcDataset = signal<Dataset | null>(null);
  hpcPathError = signal<string | null>(null);
  hpcValidating = signal(false);
  startingOllama = signal(false);
  ollamaStartError = signal<string | null>(null);
  forkSource = signal<Session | null>(null);
  forkSubmitting = signal(false);
  forkError = signal<string | null>(null);
  pythonEnvironmentChoice = signal('bundled');
  manualPythonEnvironmentPath = signal('');
  validatedPythonEnvironment = signal<PythonEnvironmentCandidate | null>(null);
  pythonEnvironmentError = signal<string | null>(null);
  pythonEnvironmentLoading = signal(false);
  forkPythonEnvironmentChoice = signal('inherit');
  forkManualPythonEnvironmentPath = signal('');
  forkValidatedPythonEnvironment = signal<PythonEnvironmentCandidate | null>(null);
  forkPythonEnvironmentError = signal<string | null>(null);
  forkPythonEnvironmentLoading = signal(false);
  forkForm: SessionForkRequest = {
    name: '',
    recovery_mode: 'smart',
    target_mode: 'interactive',
    additional_turns: 20,
    acknowledge_replay_risk: false,
  };

  // Preset flow
  presets = PRESETS;
  selectedPreset = signal<string | null>('scrna-explore');
  wizardStep = signal<'preset' | 'form' | 'confirm'>('preset');

  // Sort/filter/bulk
  sortKey = signal<SortKey>('newest');
  statusFilter = signal<StatusFilter>('all');
  searchQuery = signal('');
  selected = signal<Set<string>>(new Set());
  selectMode = signal(false);

  get pendingDeleteSession() {
    return this.sessionSvc.sessions().find((s) => s.id === this.pendingDeleteId());
  }

  form: SessionCreateRequest = {
    name: '',
    mode: 'interactive',
    run_mode: 'full_system',
    agent_system: '',
    llm_backend: '',
    model_name: '',
    ollama_model: '',
    sandbox_type: 'singularity',
    dataset_path: '',
    max_turns: 20,
    initial_prompt: '',
    memory_strategy: 'full',
    memory_working_history_size: 4,
    memory_summarization_threshold: 20,
    memory_chunk_size: 10,
    evaluator_model: { mode: 'inherit_worker' },
  };

  // Separate from `form.brief_mode`: the wizard needs a fourth state
  // ("use whatever the blueprint says") that has no representation as a
  // SessionCreateRequest value — that field is undefined/omitted for it,
  // not a fourth enum member on the wire.
  briefModeChoice: 'blueprint' | 'off' | 'context' | 'seed_item' = 'blueprint';

  onBriefModeChange(choice: 'blueprint' | 'off' | 'context' | 'seed_item'): void {
    this.briefModeChoice = choice;
    this.form.brief_mode = choice === 'blueprint' ? undefined : choice;
  }

  availableBackends = computed(() =>
    this.configSvc
      .backends()
      .filter((b) => b.available || b.id === 'ollama' || b.id === 'openrouter'),
  );

  selectedBackend = computed(
    () => this.configSvc.backends().find((b) => b.id === this.form.llm_backend) ?? null,
  );

  selectedOllamaStatus = computed(() =>
    this.form.llm_backend === 'ollama' ? this.configSvc.ollamaModels() : null,
  );

  selectedDatasetInfo = computed(() => {
    return this.datasets().find((d) => d.path === this.form.dataset_path) ?? this.hpcDataset();
  });

  filteredSortedSessions = computed(() => {
    const q = this.searchQuery().trim().toLowerCase();
    const status = this.statusFilter();
    const sort = this.sortKey();
    let items = this.sessionSvc.sessions().slice();
    if (status !== 'all') items = items.filter((s) => s.status === status);
    if (q) {
      items = items.filter(
        (s) =>
          s.id.toLowerCase().includes(q) ||
          s.name.toLowerCase().includes(q) ||
          s.agent_system.toLowerCase().includes(q) ||
          s.llm_backend.toLowerCase().includes(q) ||
          (s.resolved_model?.model ?? '').toLowerCase().includes(q) ||
          (s.current_agent ?? '').toLowerCase().includes(q),
      );
    }
    switch (sort) {
      case 'newest':
        items.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
        break;
      case 'oldest':
        items.sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
        break;
      case 'status':
        items.sort((a, b) => a.status.localeCompare(b.status));
        break;
      case 'turns':
        items.sort((a, b) => b.current_turn - a.current_turn);
        break;
    }
    return items;
  });

  statusCounts = computed(() => {
    const counts: Record<StatusFilter, number> = {
      all: 0,
      running: 0,
      idle: 0,
      stopped: 0,
      error: 0,
      initializing: 0,
      recovering: 0,
    };
    for (const s of this.sessionSvc.sessions()) {
      counts.all++;
      (counts as any)[s.status]++;
    }
    return counts;
  });

  ngOnInit(): void {
    this.sessionSvc.loadSessions().subscribe();
    this.configSvc.loadAll().subscribe(() => this.applyDefaults());
    this.datasetSvc.getDatasets().subscribe((d) => this.datasets.set(d));
  }

  private applyDefaults(): void {
    const bp =
      this.configSvc.blueprints().find((b) => b.name === DEFAULT_AGENT_SYSTEM) ??
      this.configSvc.blueprints()[0];
    if (bp && !this.form.agent_system) this.form.agent_system = bp.name;
    const be =
      this.availableBackends().find((backend) => backend.available) ??
      this.availableBackends().find((backend) => backend.id === 'ollama');
    if (be && !this.form.llm_backend) this.form.llm_backend = be.id;
    this.form.memory_strategy = 'full';
    this.form.memory_working_history_size = 4;
    this.form.memory_summarization_threshold = 20;
    this.form.memory_chunk_size = 10;
    this.briefModeChoice = 'blueprint';
    this.form.brief_mode = undefined;
    this.applyOllamaDefaultModel();
  }

  openCreate(): void {
    this.createError.set(null);
    this.wizardStep.set('preset');
    this.selectedPreset.set('scrna-explore');
    this.pythonEnvironmentChoice.set('bundled');
    this.form.python_environment_path = undefined;
    this.manualPythonEnvironmentPath.set('');
    this.validatedPythonEnvironment.set(null);
    this.pythonEnvironmentError.set(null);
    this.showCreateDialog.set(true);
  }

  cancelCreate(): void {
    this.showCreateDialog.set(false);
  }

  choosePreset(id: string): void {
    this.selectedPreset.set(id);
    const preset = PRESETS.find((p) => p.id === id);
    if (!preset) return;
    // Apply preset values on top of defaults
    this.applyDefaults();
    if (preset.values.mode) this.form.mode = preset.values.mode;
    if (preset.values.run_mode) this.form.run_mode = preset.values.run_mode;
    if (preset.values.agent_system) this.form.agent_system = preset.values.agent_system;
    if (preset.values.sandbox_type) this.form.sandbox_type = preset.values.sandbox_type;
    if (preset.values.max_turns) this.form.max_turns = preset.values.max_turns;
    if (preset.values.initial_prompt !== undefined)
      this.form.initial_prompt = preset.values.initial_prompt;
    this.wizardStep.set('form');
  }

  goToConfirm(): void {
    const err = this.validateForm();
    if (err) {
      this.createError.set(err);
      return;
    }
    this.createError.set(null);
    this.wizardStep.set('confirm');
  }

  backToForm(): void {
    this.wizardStep.set('form');
  }

  private validateForm(): string | null {
    if (!this.form.name?.trim()) {
      return 'Enter a session name.';
    }
    if (this.form.llm_backend === 'ollama') {
      const ollama = this.configSvc.ollamaModels();
      if (ollama?.status === 'no_models') {
        return ollama.suggested_fix ?? 'Download an Ollama model before creating a session.';
      }
      if (ollama?.status === 'not_installed' || ollama?.status === 'unreachable') {
        return ollama.suggested_fix ?? ollama.message ?? 'Ollama is not reachable.';
      }
      this.form.ollama_model =
        this.form.ollama_model || (ollama?.models.length ? ollama.default_model : undefined);
    } else {
      this.form.ollama_model = undefined;
    }
    if (this.form.llm_backend === 'openrouter' && this.selectedBackend()?.available === false) {
      return (
        this.selectedBackend()?.suggested_fix ??
        'Configure an OpenRouter API key in Settings before creating this session.'
      );
    }
    if (this.form.llm_backend === 'openrouter' && !this.form.model_name) {
      return 'Select an OpenRouter model before creating a session.';
    }
    if (
      this.form.evaluator_model?.mode === 'explicit' &&
      (!this.form.evaluator_model.llm_backend || !this.form.evaluator_model.model_name)
    ) {
      return 'Select an evaluator backend and enter an exact evaluator model identifier.';
    }
    if (this.datasetSource() === 'hpc' && !this.hpcDataset()) {
      return 'Validate the HPC dataset path before creating a session.';
    }
    if (
      this.pythonEnvironmentChoice() === 'manual' &&
      !this.validatedPythonEnvironment()
    ) {
      return 'Validate the host Python environment path before creating a session.';
    }
    if (!this.form.agent_system || !this.form.llm_backend || !this.form.dataset_path) {
      return 'Blueprint, backend, and dataset are required.';
    }
    return null;
  }

  submitCreate(): void {
    const err = this.validateForm();
    if (err) {
      this.createError.set(err);
      this.wizardStep.set('form');
      return;
    }
    const request: SessionCreateRequest = {
      ...this.form,
      name: this.form.name?.trim(),
      max_turns: this.form.mode === 'auto' ? this.form.max_turns : undefined,
    };
    this.creating.set(true);
    this.createError.set(null);
    this.sessionSvc.createSession(request).subscribe({
      next: (s) => {
        this.creating.set(false);
        this.showCreateDialog.set(false);
        this.router.navigate(['/session', s.id]);
      },
      error: (err) => {
        this.creating.set(false);
        this.createError.set(err?.error?.detail ?? 'Failed to create session.');
        this.wizardStep.set('form');
      },
    });
  }

  onBackendChange(backend: string): void {
    this.form.llm_backend = backend;
    this.createError.set(null);
    if (backend === 'ollama') {
      this.ollamaStartError.set(null);
      this.configSvc.getOllamaModels().subscribe({
        next: () => this.applyOllamaDefaultModel(),
        error: () => this.applyOllamaDefaultModel(),
      });
    } else if (backend === 'openrouter') {
      if (this.selectedBackend()?.available) {
        this.loadOpenRouterModels();
      } else {
        this.openRouterError.set(null);
      }
    }
  }

  onPythonEnvironmentChange(value: string): void {
    this.pythonEnvironmentChoice.set(value);
    this.pythonEnvironmentError.set(null);
    this.validatedPythonEnvironment.set(null);
    if (value === 'bundled' || value === 'manual') {
      this.form.python_environment_path = undefined;
    } else {
      this.form.python_environment_path = value;
    }
  }

  onManualPythonEnvironmentInput(value: string): void {
    this.manualPythonEnvironmentPath.set(value);
    this.validatedPythonEnvironment.set(null);
    this.form.python_environment_path = undefined;
    this.pythonEnvironmentError.set(null);
  }

  validateManualPythonEnvironment(): void {
    const path = this.manualPythonEnvironmentPath().trim();
    if (!path) {
      this.pythonEnvironmentError.set('Enter an absolute environment prefix.');
      return;
    }
    this.pythonEnvironmentLoading.set(true);
    this.pythonEnvironmentError.set(null);
    this.configSvc.validatePythonEnvironment(path).subscribe({
      next: (candidate) => {
        this.pythonEnvironmentLoading.set(false);
        this.validatedPythonEnvironment.set(candidate);
        this.form.python_environment_path = candidate.path;
      },
      error: (err) => {
        this.pythonEnvironmentLoading.set(false);
        this.validatedPythonEnvironment.set(null);
        this.form.python_environment_path = undefined;
        this.pythonEnvironmentError.set(
          err?.error?.detail ?? 'The environment prefix is not usable.',
        );
      },
    });
  }

  refreshPythonEnvironments(): void {
    this.pythonEnvironmentLoading.set(true);
    this.pythonEnvironmentError.set(null);
    this.configSvc.getPythonEnvironments().subscribe({
      next: () => this.pythonEnvironmentLoading.set(false),
      error: () => {
        this.pythonEnvironmentLoading.set(false);
        this.pythonEnvironmentError.set('Could not query environments on the server host.');
      },
    });
  }

  loadOpenRouterModels(refresh = false): void {
    this.loadingOpenRouter.set(true);
    this.openRouterError.set(null);
    this.configSvc.getOpenRouterModels(refresh).subscribe({
      next: (catalogue) => {
        this.loadingOpenRouter.set(false);
        if (!this.form.model_name && catalogue.models.length) {
          this.form.model_name = catalogue.models[0].canonical_slug;
        }
      },
      error: (err) => {
        this.loadingOpenRouter.set(false);
        this.openRouterError.set(err?.error?.detail ?? 'Unable to load OpenRouter models.');
      },
    });
  }

  startOllama(): void {
    this.startingOllama.set(true);
    this.ollamaStartError.set(null);
    this.configSvc.startOllama().subscribe({
      next: () => {
        this.startingOllama.set(false);
        this.applyOllamaDefaultModel();
      },
      error: (err) => {
        this.startingOllama.set(false);
        const detail = err?.error?.detail;
        this.ollamaStartError.set(
          detail?.suggested_fix ??
            detail?.message ??
            err?.error?.detail ??
            'Unable to start Ollama.',
        );
      },
    });
  }

  applyOllamaDefaultModel(): void {
    if (this.form.llm_backend !== 'ollama') return;
    const ollama = this.configSvc.ollamaModels();
    if (!ollama) return;
    if (this.form.ollama_model && ollama.models.includes(this.form.ollama_model)) return;
    this.form.ollama_model = ollama.models.length ? ollama.default_model || ollama.models[0] : '';
  }

  openSession(id: string): void {
    if (this.selectMode()) {
      this.toggleSelected(id);
      return;
    }
    this.router.navigate(['/session', id]);
  }

  requestDelete(id: string, event: Event): void {
    event.stopPropagation();
    this.pendingDeleteId.set(id);
  }

  confirmDelete(): void {
    const id = this.pendingDeleteId();
    if (!id) return;
    this.sessionSvc.deleteSession(id).subscribe();
    this.pendingDeleteId.set(null);
  }

  cancelDelete(): void {
    this.pendingDeleteId.set(null);
  }

  duplicateSession(s: Session, event: Event): void {
    event.stopPropagation();
    this.forkError.set(null);
    this.forkPythonEnvironmentChoice.set('inherit');
    this.forkManualPythonEnvironmentPath.set('');
    this.forkValidatedPythonEnvironment.set(null);
    this.forkPythonEnvironmentError.set(null);
    this.forkForm = {
      name: `${s.name || s.id.slice(0, 8)} fork`,
      llm_backend: s.llm_backend,
      model_name: s.llm_backend === 'openrouter' ? s.resolved_model?.model ?? '' : undefined,
      ollama_model: s.llm_backend === 'ollama' ? s.resolved_model?.model ?? '' : undefined,
      recovery_mode: 'smart',
      target_mode: s.mode === 'auto' ? 'interactive' : s.mode,
      additional_turns: 20,
      acknowledge_replay_risk: false,
    };
    this.forkSource.set(s);
    if (s.llm_backend === 'openrouter' && !this.configSvc.openRouterCatalogue()) {
      this.loadOpenRouterModels();
    }
  }

  cancelFork(): void {
    if (!this.forkSubmitting()) this.forkSource.set(null);
  }

  submitFork(): void {
    const source = this.forkSource();
    if (!source || !this.forkForm.name.trim()) {
      this.forkError.set('A name is required for the fork.');
      return;
    }
    if (this.forkForm.recovery_mode === 'literal_replay' && !this.forkForm.acknowledge_replay_risk) {
      this.forkError.set('Acknowledge the literal replay side-effect warning.');
      return;
    }
    if (this.forkForm.target_mode === 'auto' && !(this.forkForm.additional_turns! > 0)) {
      this.forkError.set('Enter an additional-turn budget for Auto mode.');
      return;
    }
    if (
      this.forkPythonEnvironmentChoice() === 'manual' &&
      !this.forkValidatedPythonEnvironment()
    ) {
      this.forkError.set('Validate the host Python environment path before creating the fork.');
      return;
    }
    const forkTab = reserveNewTab();
    this.forkSubmitting.set(true);
    this.forkError.set(null);
    this.sessionSvc.forkSession(source.id, { ...this.forkForm, name: this.forkForm.name.trim() }).subscribe({
      next: child => {
        this.forkSubmitting.set(false);
        this.forkSource.set(null);
        if (!navigateTabToSession(forkTab, child.id)) {
          this.toasts.show({
            kind: 'warn',
            title: 'New tab blocked',
            detail: 'The fork was opened in this tab instead.',
            ttlMs: 6000,
          });
          void this.router.navigate(['/session', child.id]);
        }
      },
      error: err => {
        forkTab?.close();
        this.forkSubmitting.set(false);
        this.forkError.set(err?.error?.detail ?? 'Unable to fork this session.');
      },
    });
  }

  onForkBackendChange(backend: string): void {
    this.forkForm.llm_backend = backend;
    this.forkForm.model_name = undefined;
    this.forkForm.ollama_model = undefined;
    if (backend === 'openrouter') {
      this.configSvc.getOpenRouterModels().subscribe(catalogue => {
        this.forkForm.model_name = catalogue.models[0]?.canonical_slug;
      });
    } else if (backend === 'ollama') {
      this.configSvc.getOllamaModels().subscribe(result => {
        this.forkForm.ollama_model = result.default_model || result.models[0];
      });
    }
  }

  onForkPythonEnvironmentChange(value: string): void {
    this.forkPythonEnvironmentChoice.set(value);
    this.forkPythonEnvironmentError.set(null);
    this.forkValidatedPythonEnvironment.set(null);
    if (value === 'inherit' || value === 'manual') {
      this.forkForm.python_environment_path = undefined;
    } else if (value === 'bundled') {
      this.forkForm.python_environment_path = null;
    } else {
      this.forkForm.python_environment_path = value;
    }
  }

  onForkManualPythonEnvironmentInput(value: string): void {
    this.forkManualPythonEnvironmentPath.set(value);
    this.forkValidatedPythonEnvironment.set(null);
    this.forkForm.python_environment_path = undefined;
    this.forkPythonEnvironmentError.set(null);
  }

  validateForkManualPythonEnvironment(): void {
    const path = this.forkManualPythonEnvironmentPath().trim();
    if (!path) {
      this.forkPythonEnvironmentError.set('Enter an absolute environment prefix.');
      return;
    }
    this.forkPythonEnvironmentLoading.set(true);
    this.forkPythonEnvironmentError.set(null);
    this.configSvc.validatePythonEnvironment(path).subscribe({
      next: (candidate) => {
        this.forkPythonEnvironmentLoading.set(false);
        this.forkValidatedPythonEnvironment.set(candidate);
        this.forkForm.python_environment_path = candidate.path;
      },
      error: (err) => {
        this.forkPythonEnvironmentLoading.set(false);
        this.forkValidatedPythonEnvironment.set(null);
        this.forkForm.python_environment_path = undefined;
        this.forkPythonEnvironmentError.set(
          err?.error?.detail ?? 'The environment prefix is not usable.',
        );
      },
    });
  }

  refreshForkPythonEnvironments(): void {
    this.forkPythonEnvironmentLoading.set(true);
    this.forkPythonEnvironmentError.set(null);
    this.configSvc.getPythonEnvironments().subscribe({
      next: () => this.forkPythonEnvironmentLoading.set(false),
      error: () => {
        this.forkPythonEnvironmentLoading.set(false);
        this.forkPythonEnvironmentError.set('Could not query environments on the server host.');
      },
    });
  }

  toggleSelectMode(): void {
    this.selectMode.update((v) => !v);
    if (!this.selectMode()) this.selected.set(new Set());
  }

  toggleSelected(id: string): void {
    this.selected.update((set) => {
      const next = new Set(set);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  isSelected(id: string): boolean {
    return this.selected().has(id);
  }

  selectAll(): void {
    this.selected.set(new Set(this.filteredSortedSessions().map((s) => s.id)));
  }

  clearSelection(): void {
    this.selected.set(new Set());
  }

  requestBulkDelete(): void {
    if (this.selected().size === 0) return;
    this.pendingBulkDelete.set(true);
  }

  confirmBulkDelete(): void {
    const ids = Array.from(this.selected());
    ids.forEach((id) => this.sessionSvc.deleteSession(id).subscribe());
    this.selected.set(new Set());
    this.pendingBulkDelete.set(false);
    this.toasts.show({
      kind: 'success',
      title: `Deleted ${ids.length} session${ids.length === 1 ? '' : 's'}`,
      ttlMs: 3500,
    });
  }

  cancelBulkDelete(): void {
    this.pendingBulkDelete.set(false);
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    this.datasetSvc.uploadDataset(file).subscribe((ev) => {
      if (ev.dataset) {
        this.datasets.update((d) => [...d, ev.dataset!]);
        this.form.dataset_path = ev.dataset!.path;
        this.datasetSource.set('existing');
      }
    });
  }

  onDatasetSourceChange(source: DatasetSource): void {
    this.datasetSource.set(source);
    this.createError.set(null);
    if (source === 'hpc') {
      this.form.dataset_path = this.hpcDataset()?.path ?? '';
    } else {
      this.form.dataset_path = '';
      this.hpcPathError.set(null);
      this.hpcValidating.set(false);
    }
  }

  onHpcPathInput(value: string): void {
    this.hpcDatasetPath.set(value);
    this.hpcDataset.set(null);
    this.hpcPathError.set(null);
    this.form.dataset_path = '';
  }

  validateHpcPath(): void {
    const path = this.hpcDatasetPath().trim();
    if (!path) {
      this.hpcPathError.set('Enter an absolute HPC path to a .h5ad file.');
      return;
    }
    this.hpcValidating.set(true);
    this.hpcPathError.set(null);
    this.datasetSvc.validateHpcPath(path).subscribe({
      next: (dataset) => {
        this.hpcValidating.set(false);
        this.hpcDataset.set(dataset);
        this.form.dataset_path = dataset.path;
      },
      error: (err) => {
        this.hpcValidating.set(false);
        this.hpcDataset.set(null);
        this.form.dataset_path = '';
        this.hpcPathError.set(err?.error?.detail ?? 'Path is not a readable .h5ad file.');
      },
    });
  }

  formatSize(bytes: number): string {
    if (bytes > 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
    if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' KB';
  }
}
