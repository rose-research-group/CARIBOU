import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import {
  PresetExecutor,
  PresetProfile,
  PresetProvider,
  PresetSummary,
} from '../../../core/models/experiment-control.model';
import { Dataset } from '../../../core/models/session.model';
import { TooltipDirective } from '../../../shared/directives/tooltip.directive';

@Component({
  selector: 'app-confirm-step',
  standalone: true,
  imports: [CommonModule, TooltipDirective],
  templateUrl: './confirm-step.component.html',
  styleUrls: ['./confirm-step.component.scss'],
})
export class ConfirmStepComponent {
  @Input() selectedDataset: Dataset | null = null;
  @Input() preset: PresetSummary | null = null;
  @Input() profile: PresetProfile = 'fast';
  @Input() maxTurns = 10;
  @Input() provider: PresetProvider = 'openai';
  @Input() modelName = '';
  @Input() evaluatorProvider: PresetProvider = 'openai';
  @Input() evaluatorModelName = '';
  @Input() openrouterEndpoint = '';
  @Input() executor: PresetExecutor = 'slurm';
  @Input() owner = '';
  @Input() reviewer = '';
  @Input() specHash = '';
  @Input() planHash = '';

  get cpuCores(): number {
    return this.preset?.resource_profiles[this.profile].cpu_cores ?? 0;
  }

  get memoryGB(): number {
    return (this.preset?.resource_profiles[this.profile].memory_bytes ?? 0) / 1e9;
  }
}
