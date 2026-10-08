import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { map, switchMap } from 'rxjs';
import {
  DEEPSEEK_MODELS,
  PresetExecutor,
  PresetProfile,
  PresetProvider,
  PresetResolveRequest,
  PresetSummary,
} from '../../core/models/experiment-control.model';
import { Dataset } from '../../core/models/session.model';
import { ConfigService } from '../../core/services/config.service';
import { DatasetService } from '../../core/services/dataset.service';
import { ExperimentControlService } from '../../core/services/experiment-control.service';
import { TooltipDirective } from '../../shared/directives/tooltip.directive';
import { ConfirmStepComponent } from './steps/confirm-step.component';
import { DatasetStepComponent } from './steps/dataset-step.component';
import { PresetStepComponent } from './steps/preset-step.component';
import { ResourcesStepComponent } from './steps/resources-step.component';

type Step = 'dataset' | 'preset' | 'resources' | 'confirm';

@Component({
  selector: 'app-wizard',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RouterLink,
    DatasetStepComponent,
    PresetStepComponent,
    ResourcesStepComponent,
    ConfirmStepComponent,
    TooltipDirective,
  ],
  templateUrl: './wizard.component.html',
  styleUrls: ['./wizard.component.scss'],
})
export class WizardComponent implements OnInit {
  private router = inject(Router);
  private control = inject(ExperimentControlService);
  private datasetService = inject(DatasetService);
  configService = inject(ConfigService);

  presets = signal<PresetSummary[]>([]);
  datasets = signal<Dataset[]>([]);
  currentStep = signal<Step>('dataset');
  selectedDataset = signal<Dataset | null>(null);
  selectedPreset = signal<string | null>(null);
  profile = signal<PresetProfile>('fast');
  maxTurns = signal(10);
  provider = signal<PresetProvider>('openai');
  modelName = signal('gpt-4.1-2025-04-14');
  evaluatorProvider = signal<PresetProvider>('openai');
  evaluatorModelName = signal('gpt-4.1-2025-04-14');
  openrouterEndpoint = signal('');
  executor = signal<PresetExecutor>('slurm');
  owner = signal('web-operator');
  reviewer = signal('operator-review-required');
  resolvedSpecification = signal<Record<string, unknown> | null>(null);
  specHash = signal('');
  planHash = signal('');
  idempotencyKey = signal('');
  error = signal<string | null>(null);
  busy = signal(false);

  tokenConfigured = this.control.hasAccessToken;
  selectedPresetDefinition = computed(() => {
    const id = this.selectedPreset();
    return this.presets().find((preset) => preset.id === id) ?? null;
  });
  currentStepNumber = computed(() => {
    const steps: Step[] = ['dataset', 'preset', 'resources', 'confirm'];
    return steps.indexOf(this.currentStep()) + 1;
  });
  nextButtonLabel = computed(() => {
    switch (this.currentStep()) {
      case 'dataset':
        return 'Choose analysis →';
      case 'preset':
        return 'Configure run →';
      case 'resources':
        return 'Review plan →';
      default:
        return 'Submit experiment';
    }
  });

  ngOnInit(): void {
    this.loadDatasets();
    if (this.tokenConfigured()) {
      this.loadPresets();
    } else {
      this.error.set('Configure the control API bearer token on the Experiments page first.');
    }
  }

  loadPresets(): void {
    this.control.presets().subscribe({
      next: (response) => this.presets.set(response.data.presets),
      error: (error) => this.setError(error),
    });
  }

  loadDatasets(): void {
    this.datasetService.getDatasets().subscribe({
      next: (datasets) => this.datasets.set(datasets),
      error: (error) => this.setError(error),
    });
  }

  selectDataset(dataset: Dataset): void {
    this.selectedDataset.set(dataset);
    this.invalidatePreparedSpec();
  }

  selectPreset(id: string): void {
    const preset = this.presets().find((candidate) => candidate.id === id);
    this.selectedPreset.set(id);
    if (preset) {
      this.profile.set(preset.default_profile);
      this.maxTurns.set(preset.default_max_turns);
    }
    this.invalidatePreparedSpec();
  }

  updateProfile(value: PresetProfile): void {
    this.profile.set(value);
    this.invalidatePreparedSpec();
  }

  updateMaxTurns(value: number): void {
    this.maxTurns.set(Number(value));
    this.invalidatePreparedSpec();
  }

  updateProvider(value: PresetProvider): void {
    this.provider.set(value);
    if (value === 'deepseek') {
      const selectedModel = this.modelName();
      if (!DEEPSEEK_MODELS.some((model) => model.id === selectedModel)) {
        this.modelName.set(DEEPSEEK_MODELS[0].id);
      }
    } else if (value === 'openrouter') {
      this.openrouterEndpoint.set('');
      this.configService.getOpenRouterModels().subscribe({
        next: (catalogue) => {
          const first = catalogue.models[0];
          if (first) this.updateModelName(first.canonical_slug);
        },
        error: (error) => this.setError(error),
      });
    } else if (
      DEEPSEEK_MODELS.some((model) => model.id === this.modelName()) ||
      this.modelName().includes('/')
    ) {
      this.modelName.set('gpt-4.1-2025-04-14');
    }
    this.invalidatePreparedSpec();
  }

  updateModelName(value: string): void {
    this.modelName.set(value);
    if (this.provider() === 'openrouter') {
      this.openrouterEndpoint.set('');
    }
    this.invalidatePreparedSpec();
  }

  updateOpenRouterEndpoint(value: string): void {
    this.openrouterEndpoint.set(value);
    this.invalidatePreparedSpec();
  }

  updateExecutor(value: PresetExecutor): void {
    this.executor.set(value);
    this.invalidatePreparedSpec();
  }

  updateOwner(value: string): void {
    this.owner.set(value);
    this.invalidatePreparedSpec();
  }

  updateReviewer(value: string): void {
    this.reviewer.set(value);
    this.invalidatePreparedSpec();
  }

  nextStep(): void {
    this.error.set(null);
    if (this.currentStep() === 'dataset') {
      if (!this.selectedDataset()) {
        this.error.set('Select a dataset before continuing.');
        return;
      }
      this.currentStep.set('preset');
      return;
    }
    if (this.currentStep() === 'preset') {
      if (!this.selectedPresetDefinition()) {
        this.error.set('Select a preset before continuing.');
        return;
      }
      this.currentStep.set('resources');
      return;
    }
    if (this.currentStep() === 'resources') {
      this.prepareConfirmation();
    }
  }

  prevStep(): void {
    const steps: Step[] = ['dataset', 'preset', 'resources', 'confirm'];
    const currentIndex = steps.indexOf(this.currentStep());
    if (currentIndex > 0) {
      if (this.currentStep() === 'confirm') this.invalidatePreparedSpec();
      this.currentStep.set(steps[currentIndex - 1]);
    }
  }

  canNavigateTo(step: Step): boolean {
    if (step === 'dataset') return true;
    if (step === 'preset') return this.selectedDataset() !== null;
    if (step === 'resources') return this.selectedPresetDefinition() !== null;
    return this.resolvedSpecification() !== null;
  }

  goToStep(step: Step): void {
    if (this.busy() || !this.canNavigateTo(step) || step === this.currentStep()) return;
    if (this.currentStep() === 'confirm') this.invalidatePreparedSpec();
    this.currentStep.set(step);
    this.error.set(null);
  }

  submit(): void {
    const specification = this.resolvedSpecification();
    const planHash = this.planHash();
    const idempotencyKey = this.idempotencyKey();
    if (!specification || !planHash || !idempotencyKey) {
      this.error.set('Resolve and plan the preset before submitting it.');
      return;
    }
    this.busy.set(true);
    this.error.set(null);
    this.control.submit(specification, idempotencyKey, planHash).subscribe({
      next: (response) => {
        const runId = response.data.run_ids[0];
        if (runId) {
          try {
            localStorage.setItem('caribou:control:last-run:v1', runId);
          } catch {
            // Navigation still succeeds when browser storage is unavailable.
          }
        }
        this.busy.set(false);
        void this.router.navigate(['/experiments']);
      },
      error: (error) => {
        this.busy.set(false);
        this.setError(error);
      },
    });
  }

  private prepareConfirmation(): void {
    const dataset = this.selectedDataset();
    const preset = this.selectedPresetDefinition();
    const maxTurns = this.maxTurns();
    if (!dataset || !preset) {
      this.error.set('Dataset and preset selections are required.');
      return;
    }
    if (!Number.isInteger(maxTurns) || maxTurns < 1 || maxTurns > preset.maximum_max_turns) {
      this.error.set(`Maximum turns must be between 1 and ${preset.maximum_max_turns}.`);
      return;
    }
    if (!this.modelName().trim() || !this.evaluatorModelName().trim() || !this.owner().trim() || !this.reviewer().trim()) {
      this.error.set('Worker model, evaluator model, owner, and reviewer are required.');
      return;
    }
    if (this.provider() === 'openrouter' && !this.openrouterEndpoint().trim()) {
      this.error.set('Select an OpenRouter provider endpoint for this reproducible experiment.');
      return;
    }
    const request: PresetResolveRequest = {
      dataset_path: dataset.path,
      model_provider: this.provider(),
      model_name: this.modelName().trim(),
      openrouter_endpoint:
        this.provider() === 'openrouter' ? this.openrouterEndpoint().trim() : undefined,
      profile: this.profile(),
      max_turns: maxTurns,
      executor: this.executor(),
      owner: this.owner().trim(),
      reviewer: this.reviewer().trim(),
      evaluator_provider: this.evaluatorProvider(),
      evaluator_model_name: this.evaluatorModelName().trim(),
    };

    this.busy.set(true);
    this.error.set(null);
    this.control
      .resolvePreset(preset.id, request)
      .pipe(
        switchMap((resolution) =>
          this.control
            .plan(resolution.data.specification)
            .pipe(map((plan) => ({ resolution, plan }))),
        ),
      )
      .subscribe({
        next: ({ resolution, plan }) => {
          const currentPlanHash = plan.data['plan_hash'];
          if (typeof currentPlanHash !== 'string' || !currentPlanHash) {
            this.busy.set(false);
            this.error.set('The server returned a plan without an integrity hash.');
            return;
          }
          this.resolvedSpecification.set(resolution.data.specification);
          this.specHash.set(resolution.data.spec_hash);
          this.planHash.set(currentPlanHash);
          this.idempotencyKey.set(`wizard-${resolution.object.id}`);
          this.currentStep.set('confirm');
          this.busy.set(false);
        },
        error: (error) => {
          this.busy.set(false);
          this.setError(error);
        },
      });
  }

  private invalidatePreparedSpec(): void {
    this.resolvedSpecification.set(null);
    this.specHash.set('');
    this.planHash.set('');
    this.idempotencyKey.set('');
    this.error.set(null);
  }

  private setError(error: unknown): void {
    if (error instanceof HttpErrorResponse) {
      const body = error.error;
      const message = body?.error?.message ?? body?.detail?.error?.message ?? body?.detail;
      if (typeof message === 'string' && message) {
        this.error.set(message);
        return;
      }
    }
    this.error.set(error instanceof Error ? error.message : 'The request could not be completed.');
  }
}
