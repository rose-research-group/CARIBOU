import { CommonModule } from '@angular/common';
import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  DEEPSEEK_MODELS,
  PresetExecutor,
  PresetProfile,
  PresetProvider,
  PresetSummary,
} from '../../../core/models/experiment-control.model';
import { TooltipDirective } from '../../../shared/directives/tooltip.directive';
import { OpenRouterEndpoint } from '../../../core/models/session.model';
import { ConfigService } from '../../../core/services/config.service';

@Component({
  selector: 'app-resources-step',
  standalone: true,
  imports: [CommonModule, FormsModule, TooltipDirective],
  templateUrl: './resources-step.component.html',
  styleUrls: ['./resources-step.component.scss'],
})
export class ResourcesStepComponent implements OnChanges {
  readonly config = inject(ConfigService);
  readonly deepseekModels = DEEPSEEK_MODELS;
  readonly openrouterEndpoints = signal<OpenRouterEndpoint[]>([]);
  readonly loadingOpenrouterEndpoints = signal(false);
  readonly openrouterError = signal<string | null>(null);

  @Input() preset: PresetSummary | null = null;
  @Input() profile: PresetProfile = 'fast';
  @Input() maxTurns = 10;
  @Input() provider: PresetProvider = 'openai';
  @Input() modelName = 'gpt-4.1-2025-04-14';
  @Input() evaluatorProvider: PresetProvider = 'openai';
  @Input() evaluatorModelName = 'gpt-4.1-2025-04-14';
  @Input() openrouterEndpoint = '';
  @Input() executor: PresetExecutor = 'slurm';
  @Input() owner = '';
  @Input() reviewer = '';

  @Output() profileChange = new EventEmitter<PresetProfile>();
  @Output() maxTurnsChange = new EventEmitter<number>();
  @Output() providerChange = new EventEmitter<PresetProvider>();
  @Output() modelNameChange = new EventEmitter<string>();
  @Output() evaluatorProviderChange = new EventEmitter<PresetProvider>();
  @Output() evaluatorModelNameChange = new EventEmitter<string>();
  @Output() openrouterEndpointChange = new EventEmitter<string>();
  @Output() executorChange = new EventEmitter<PresetExecutor>();
  @Output() ownerChange = new EventEmitter<string>();
  @Output() reviewerChange = new EventEmitter<string>();

  ngOnChanges(changes: SimpleChanges): void {
    if (
      (changes['modelName'] || changes['provider']) &&
      this.provider === 'openrouter' &&
      this.modelName
    ) {
      this.loadEndpoints();
    }
  }

  loadEndpoints(): void {
    this.loadingOpenrouterEndpoints.set(true);
    this.openrouterError.set(null);
    this.config.getOpenRouterEndpoints(this.modelName).subscribe({
      next: (response) => {
        this.loadingOpenrouterEndpoints.set(false);
        this.openrouterEndpoints.set(response.endpoints);
        if (!response.endpoints.some((endpoint) => endpoint.slug === this.openrouterEndpoint)) {
          this.openrouterEndpointChange.emit('');
        }
      },
      error: (error) => {
        this.loadingOpenrouterEndpoints.set(false);
        this.openrouterEndpoints.set([]);
        this.openrouterError.set(error?.error?.detail ?? 'Unable to load provider endpoints.');
      },
    });
  }

  get cpuCores(): number {
    return this.preset?.resource_profiles[this.profile].cpu_cores ?? 0;
  }

  get memoryGB(): number {
    const bytes = this.preset?.resource_profiles[this.profile].memory_bytes ?? 0;
    return bytes / 1e9;
  }

  get wallHours(): number {
    const seconds = this.preset?.resource_profiles[this.profile].wall_seconds ?? 0;
    return seconds / 3600;
  }
}
