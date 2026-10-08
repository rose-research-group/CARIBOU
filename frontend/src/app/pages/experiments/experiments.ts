import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import {
  ControlArtifact,
  ControlEvent,
  RunRecord,
} from '../../core/models/experiment-control.model';
import { ExperimentControlService } from '../../core/services/experiment-control.service';
import { TooltipDirective } from '../../shared/directives/tooltip.directive';

const LAST_RUN_KEY = 'caribou:control:last-run:v1';
const TERMINAL_STATES = new Set(['succeeded', 'failed', 'cancelled', 'rejected', 'resumable']);
type AccessStatus = 'missing' | 'saved' | 'checking' | 'verified' | 'invalid';

@Component({
  selector: 'app-experiments',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, TooltipDirective],
  templateUrl: './experiments.html',
  styleUrl: './experiments.scss',
})
export class ExperimentsComponent implements OnInit, OnDestroy {
  control = inject(ExperimentControlService);

  specificationText = signal('');
  idempotencyKey = signal('');
  expectedPlanHash = signal('');
  tokenInput = signal('');
  monitorRunId = signal('');
  run = signal<RunRecord | null>(null);
  events = signal<ControlEvent[]>([]);
  artifacts = signal<ControlArtifact[]>([]);
  cursor = signal(0);
  busy = signal(false);
  action = signal<string | null>(null);
  error = signal<string | null>(null);
  result = signal<string | null>(null);
  verifiedCount = signal<number | null>(null);
  showCloneForm = signal(false);
  cloneEvaluatorProvider = signal('openai');
  cloneEvaluatorModel = signal('');
  cloneReason = signal('');
  tokenConfigured = this.control.hasAccessToken;
  accessStatus = signal<AccessStatus>(this.tokenConfigured() ? 'saved' : 'missing');

  accessStatusLabel = computed(() => {
    switch (this.accessStatus()) {
      case 'saved':
        return 'Saved in this tab';
      case 'checking':
        return 'Checking access…';
      case 'verified':
        return 'Access verified';
      case 'invalid':
        return 'Access not verified';
      default:
        return 'Token required';
    }
  });

  accessStatusDetail = computed(() => {
    switch (this.accessStatus()) {
      case 'saved':
        return 'A token is stored, but the server has not accepted it yet.';
      case 'checking':
        return 'Sending a read-only request to the CARIBOU experiment service.';
      case 'verified':
        return 'The server accepted this token for experiment operations.';
      case 'invalid':
        return 'Replace the token or ask the CARIBOU server operator for access.';
      default:
        return 'Ask the CARIBOU server operator for the experiment access token.';
    }
  });

  isTerminal = computed(() => {
    const current = this.run();
    return current ? TERMINAL_STATES.has(current.state) : false;
  });
  canCheckpoint = computed(() => {
    const state = this.run()?.state;
    return state === 'queued' || state === 'starting' || state === 'running';
  });
  canCancel = computed(() => !!this.run() && !this.isTerminal());
  canResume = computed(() => this.run()?.state === 'resumable' && this.run()?.resume_eligible);

  cloneExperiment(): void {
    const current = this.run();
    const model = this.cloneEvaluatorModel().trim();
    if (!current || !model || this.action() !== null) return;
    this.action.set('clone');
    this.error.set(null);
    this.control.cloneExperiment(
      current.experiment_id,
      this.cloneEvaluatorProvider(),
      model,
      this.cloneReason(),
    ).subscribe({
      next: response => {
        this.action.set(null);
        this.showCloneForm.set(false);
        this.specificationText.set(JSON.stringify(response.data['specification'], null, 2));
        this.result.set(JSON.stringify(response, null, 2));
      },
      error: error => {
        this.action.set(null);
        this.setError(error);
      },
    });
  }

  private pollTimer: number | null = null;
  private polling = false;
  private activeRunId: string | null = null;

  ngOnInit(): void {
    const lastRun = localStorage.getItem(LAST_RUN_KEY);
    if (lastRun) {
      this.monitorRunId.set(lastRun);
      if (this.tokenConfigured()) this.monitor(lastRun);
    }
  }

  ngOnDestroy(): void {
    this.stopPolling();
    this.activeRunId = null;
  }

  onSpecFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    file
      .text()
      .then((text) => {
        this.specificationText.set(text);
        this.error.set(null);
      })
      .catch((error) => this.setError(error));
  }

  saveAccessToken(): void {
    const token = this.tokenInput().trim();
    if (!token) {
      this.error.set('Enter the control API bearer token.');
      return;
    }
    this.control.setAccessToken(token);
    this.tokenInput.set('');
    this.error.set(null);
    this.verifyAccess();
  }

  verifyAccess(): void {
    if (!this.tokenConfigured()) {
      this.accessStatus.set('missing');
      this.error.set('Paste the experiment access token before checking access.');
      return;
    }
    this.accessStatus.set('checking');
    this.error.set(null);
    this.control.schema().subscribe({
      next: () => this.accessStatus.set('verified'),
      error: (error) => {
        this.accessStatus.set('invalid');
        if (error instanceof HttpErrorResponse && error.status === 401) {
          this.error.set(
            'The CARIBOU server rejected this experiment access token. Check it for typing errors or request the current token from the server operator.',
          );
          return;
        }
        if (error instanceof HttpErrorResponse && error.status === 503) {
          this.error.set(
            "Experiment control is disabled on this server. The server operator should run 'caribou config get-control-token' and restart with 'caribou serve'.",
          );
          return;
        }
        this.setError(error);
      },
    });
  }

  clearAccessToken(): void {
    this.stopPolling();
    this.control.clearAccessToken();
    this.tokenInput.set('');
    this.accessStatus.set('missing');
    this.error.set(null);
  }

  showSchema(): void {
    this.runAction('schema', (callback) => {
      this.control.schema().subscribe({
        next: (response) => callback(response.data.schema),
        error: (error) => this.failAction(error),
      });
    });
  }

  validateSpec(): void {
    const spec = this.parsedSpec();
    if (spec === null) return;
    this.runAction('validate', (callback) => {
      this.control.validate(spec).subscribe({
        next: (response) => callback(response),
        error: (error) => this.failAction(error),
      });
    });
  }

  planSpec(): void {
    const spec = this.parsedSpec();
    if (spec === null) return;
    this.runAction('plan', (callback) => {
      this.control.plan(spec).subscribe({
        next: (response) => {
          const planHash = response.data['plan_hash'];
          if (typeof planHash === 'string') this.expectedPlanHash.set(planHash);
          callback(response);
        },
        error: (error) => this.failAction(error),
      });
    });
  }

  submitSpec(): void {
    const spec = this.parsedSpec();
    if (spec === null) return;
    const key = this.idempotencyKey().trim();
    if (!key) {
      this.error.set('Submission requires a stable idempotency key.');
      return;
    }
    this.runAction('submit', (callback) => {
      this.control.submit(spec, key, this.expectedPlanHash().trim() || undefined).subscribe({
        next: (response) => {
          callback(response);
          const runId = response.data.run_ids[0];
          if (runId) this.monitor(runId);
        },
        error: (error) => this.failAction(error),
      });
    });
  }

  monitorRequested(): void {
    if (!this.requireAccessToken()) return;
    const runId = this.monitorRunId().trim();
    if (!runId) {
      this.error.set('Enter a durable run ID.');
      return;
    }
    this.monitor(runId);
  }

  cancel(): void {
    if (!this.requireAccessToken()) return;
    const runId = this.run()?.run_id;
    if (!runId) return;
    this.action.set('cancel');
    this.control.cancel(runId, 'cancel requested from experiment web interface').subscribe({
      next: (response) => {
        this.run.set(response.data.run);
        this.result.set(JSON.stringify(response, null, 2));
        this.action.set(null);
      },
      error: (error) => this.failAction(error),
    });
  }

  checkpoint(): void {
    if (!this.requireAccessToken()) return;
    const runId = this.run()?.run_id;
    if (!runId) return;
    this.action.set('checkpoint');
    this.control
      .requestCheckpoint(
        runId,
        `web-checkpoint-${runId}`,
        'cooperative checkpoint requested from experiment web interface',
      )
      .subscribe({
        next: (response) => {
          this.run.set(response.data.run);
          this.result.set(JSON.stringify(response, null, 2));
          this.action.set(null);
        },
        error: (error) => this.failAction(error),
      });
  }

  resume(): void {
    if (!this.requireAccessToken()) return;
    const source = this.run();
    if (!source) return;
    this.action.set('resume');
    this.control.checkpoints(source.run_id).subscribe({
      next: (response) => {
        const checkpoints = response.data.checkpoints.slice().sort((left, right) => {
          if (left.turn !== right.turn) return right.turn - left.turn;
          return right.created_at.localeCompare(left.created_at);
        });
        const checkpoint = checkpoints[0];
        if (!checkpoint) {
          this.failAction(new Error('The source run has no complete checkpoint.'));
          return;
        }
        this.control
          .resume(
            source.run_id,
            checkpoint.checkpoint_id,
            `web-resume-${source.run_id}-${checkpoint.checkpoint_id}`,
          )
          .subscribe({
            next: (resumed) => {
              this.result.set(JSON.stringify(resumed, null, 2));
              this.action.set(null);
              this.monitor(resumed.data.child_run.run_id);
            },
            error: (error) => this.failAction(error),
          });
      },
      error: (error) => this.failAction(error),
    });
  }

  verifyArtifacts(): void {
    if (!this.requireAccessToken()) return;
    const runId = this.run()?.run_id;
    if (!runId) return;
    this.action.set('verify');
    this.control.verifyArtifacts(runId).subscribe({
      next: (response) => {
        this.verifiedCount.set(response.data.verified);
        this.result.set(JSON.stringify(response, null, 2));
        this.action.set(null);
      },
      error: (error) => this.failAction(error),
    });
  }

  downloadArtifact(artifact: ControlArtifact): void {
    if (!this.requireAccessToken()) return;
    const runId = this.run()?.run_id ?? artifact.run_id;
    this.action.set(`download:${artifact.artifact_id}`);
    this.control.downloadArtifact(runId, artifact.artifact_id).subscribe({
      next: (blob) => {
        const objectUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = objectUrl;
        link.download = artifact.filename;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
        this.action.set(null);
      },
      error: (error) => this.failAction(error),
    });
  }

  eventSummary(event: ControlEvent): string {
    const payload = event.payload;
    const reason = payload['reason'];
    const message = payload['message'];
    const toState = payload['to_state'];
    if (typeof reason === 'string') return reason;
    if (typeof message === 'string') return message;
    if (typeof toState === 'string') return `state → ${toState}`;
    return event.stage ?? event.actor;
  }

  formatBytes(value: number): string {
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
  }

  private parsedSpec(): unknown | null {
    const value = this.specificationText().trim();
    if (!value) {
      this.error.set('Paste or load a frozen ExperimentSpec JSON document.');
      return null;
    }
    try {
      return JSON.parse(value) as unknown;
    } catch (error) {
      this.setError(error);
      return null;
    }
  }

  private runAction(name: string, start: (complete: (value: unknown) => void) => void): void {
    if (!this.requireAccessToken()) return;
    this.action.set(name);
    this.error.set(null);
    this.result.set(null);
    start((value) => {
      this.result.set(JSON.stringify(value, null, 2));
      this.action.set(null);
    });
  }

  private monitor(runId: string): void {
    if (!this.requireAccessToken()) return;
    this.stopPolling();
    this.activeRunId = runId;
    this.monitorRunId.set(runId);
    this.run.set(null);
    this.events.set([]);
    this.artifacts.set([]);
    this.cursor.set(0);
    this.error.set(null);
    localStorage.setItem(LAST_RUN_KEY, runId);
    this.poll();
    this.pollTimer = window.setInterval(() => this.poll(), 2000);
  }

  private poll(): void {
    const runId = this.activeRunId;
    if (!runId || this.polling) return;
    this.polling = true;
    this.control.status(runId).subscribe({
      next: (response) => {
        if (this.activeRunId !== runId) return;
        this.run.set(response.data.run);
        this.loadEventsAndArtifacts(runId);
      },
      error: (error) => {
        this.polling = false;
        this.setError(error);
      },
    });
  }

  private loadEventsAndArtifacts(runId: string): void {
    const after = this.cursor();
    this.control.events(runId, after).subscribe({
      next: (response) => {
        if (this.activeRunId !== runId) return;
        if (response.data.events.length) {
          this.events.update((current) => [...current, ...response.data.events]);
          this.cursor.set(response.data.next_cursor);
        }
        if (response.data.has_more) {
          this.polling = false;
          this.poll();
          return;
        }
        this.control.artifacts(runId).subscribe({
          next: (artifacts) => {
            if (this.activeRunId !== runId) return;
            this.artifacts.set(artifacts.data.artifacts);
            if (this.isTerminal()) this.stopPolling();
            else this.polling = false;
          },
          error: (error) => {
            this.polling = false;
            this.setError(error);
          },
        });
      },
      error: (error) => {
        this.polling = false;
        this.setError(error);
      },
    });
  }

  private stopPolling(): void {
    if (this.pollTimer !== null) window.clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.polling = false;
  }

  private requireAccessToken(): boolean {
    if (this.tokenConfigured()) return true;
    this.error.set('Configure the experiment control bearer token first.');
    return false;
  }

  private failAction(error: unknown): void {
    this.action.set(null);
    this.setError(error);
  }

  private setError(error: unknown): void {
    if (error instanceof HttpErrorResponse) {
      const payload = error.error as {
        error?: { code?: string; message?: string; retryable?: boolean };
        detail?: string;
      } | null;
      const code = payload?.error?.code;
      const message = payload?.error?.message ?? payload?.detail ?? error.message;
      const retry = payload?.error?.retryable ? ' (retryable)' : '';
      this.error.set(`${code ? `${code}: ` : ''}${message}${retry}`);
      return;
    }
    this.error.set(error instanceof Error ? error.message : String(error));
  }
}
