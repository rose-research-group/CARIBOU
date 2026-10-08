import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { ConfigService } from '../../core/services/config.service';
import { AgentBlueprint } from '../../core/models/session.model';
import { CodeEditorComponent } from '../../shared/components/code-editor/code-editor';
import { IconComponent } from '../../shared/components/icon/icon';
import { TooltipDirective } from '../../shared/directives/tooltip.directive';
import {
  AgentConfig, AgentEntry, BlueprintContent, CommandConfig,
  CommandEntry, SaveBlueprintRequest, BriefPolicyConfig,
} from '../../core/models/blueprint.model';

interface CodeSampleInfo {
  filename: string;
  size_bytes: number;
  is_builtin: boolean;
}

@Component({
  selector: 'app-blueprint-editor',
  standalone: true,
  imports: [CommonModule, FormsModule, CodeEditorComponent, IconComponent, TooltipDirective],
  templateUrl: './blueprint-editor.html',
  styleUrl: './blueprint-editor.scss',
})
export class BlueprintEditorComponent implements OnInit {
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private http = inject(HttpClient);
  configSvc = inject(ConfigService);

  // Sidebar
  allBlueprints = signal<AgentBlueprint[]>([]);

  // Editor state
  selectedName = signal<string | null>(null);
  isPackageDefault = signal(false);
  blueprintName = signal('');
  globalPolicy = signal('');
  evaluatorAgent = signal<string | null>(null);
  workItemQcMode = signal<'optional' | 'required'>('optional');
  // Not editable in the form: held as received (object, null, or absent =
  // undefined) and sent back unchanged on save.
  briefPolicy = signal<BriefPolicyConfig | null | undefined>(undefined);
  agents = signal<AgentEntry[]>([]);

  // Tabs
  activeTab = signal<'form' | 'json'>('form');
  rawJson = signal('');
  jsonError = signal<string | null>(null);

  // Action state
  saving = signal(false);
  deleting = signal(false);
  saveResult = signal<{ ok: boolean; message: string } | null>(null);

  // Per-agent import-from-path state
  importPaths = signal<string[]>([]);
  importErrors = signal<(string | null)[]>([]);
  importing = signal<boolean[]>([]);

  // Code sample autocomplete
  availableCodeSamples = signal<CodeSampleInfo[]>([]);
  sampleDropdown = signal<{ ai: number; si: number; query: string } | null>(null);
  private _blurTimer: ReturnType<typeof setTimeout> | null = null;

  filteredSuggestions = computed(() => {
    const dd = this.sampleDropdown();
    if (!dd) return [];
    const q = dd.query.toLowerCase().trim();
    const all = this.availableCodeSamples().map(s => s.filename);
    return q ? all.filter(f => f.toLowerCase().includes(q)) : all;
  });

  // Validation
  validationErrors = computed(() => this._validate(this.agents(), this.blueprintName(), this.evaluatorAgent()));

  packageBlueprints = computed(() => this.allBlueprints().filter(b => !this._isUserBlueprint(b.name)));
  userBlueprints = computed(() => this.allBlueprints().filter(b => this._isUserBlueprint(b.name)));

  ngOnInit(): void {
    this.configSvc.loadAll().subscribe(() => {
      this.allBlueprints.set(this.configSvc.blueprints());
    });

    this.http.get<CodeSampleInfo[]>('api/config/code-samples').subscribe({
      next: (list) => this.availableCodeSamples.set(list),
      error: () => {},
    });

    const name = this.route.snapshot.paramMap.get('name');
    if (name) {
      this.loadBlueprint(name);
    }
  }

  loadBlueprint(name: string): void {
    this.saveResult.set(null);
    this.http.get<BlueprintContent>(`api/config/blueprints/${name}`).subscribe({
      next: (bp) => {
        this.selectedName.set(name);
        this.isPackageDefault.set(bp.is_package_default);
        this.blueprintName.set(bp.name);
        this.globalPolicy.set(bp.global_policy);
        this.evaluatorAgent.set(bp.evaluator_agent);
        this.workItemQcMode.set(bp.work_item_policy?.qc_mode ?? 'optional');
        this.briefPolicy.set(bp.brief_policy);
        const entries = this._toAgentEntries(bp.agents);
        this.agents.set(entries);
        this._resetImportState(entries.length);
        this.activeTab.set('form');
        this.router.navigate(['/blueprints', name], { replaceUrl: true });
      },
      error: () => {
        this.saveResult.set({ ok: false, message: `Failed to load blueprint '${name}'.` });
      },
    });
  }

  newBlueprint(): void {
    this.selectedName.set(null);
    this.isPackageDefault.set(false);
    this.blueprintName.set('');
    this.globalPolicy.set('');
    this.evaluatorAgent.set(null);
    this.workItemQcMode.set('optional');
    this.briefPolicy.set(undefined);
    this.agents.set([]);
    this._resetImportState(0);
    this.activeTab.set('form');
    this.rawJson.set('');
    this.saveResult.set(null);
    this.router.navigate(['/blueprints'], { replaceUrl: true });
  }

  useAsTemplate(): void {
    this.selectedName.set(null);
    this.isPackageDefault.set(false);
    this.blueprintName.set('');
    this.saveResult.set(null);
    this._resetImportState(this.agents().length);
  }

  downloadJson(): void {
    const json = this.activeTab() === 'json' ? this.rawJson() : this._serializeToJson();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (this.blueprintName() || 'blueprint') + '.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  switchTab(tab: 'form' | 'json'): void {
    if (tab === 'json' && this.activeTab() === 'form') {
      this.rawJson.set(this._serializeToJson());
      this.jsonError.set(null);
    } else if (tab === 'form' && this.activeTab() === 'json') {
      const err = this._applyJsonToForm(this.rawJson());
      if (err) {
        this.jsonError.set(err);
        return;
      }
      this.jsonError.set(null);
    }
    this.activeTab.set(tab);
  }

  save(): void {
    if (this.activeTab() === 'json') {
      const err = this._applyJsonToForm(this.rawJson());
      if (err) {
        this.jsonError.set(err);
        return;
      }
      this.jsonError.set(null);
    }

    const errors = this.validationErrors();
    if (errors.length) return;

    const req: SaveBlueprintRequest = {
      name: this.blueprintName(),
      global_policy: this.globalPolicy(),
      agents: this._toAgentConfigMap(this.agents()),
      evaluator_agent: this.evaluatorAgent(),
      work_item_policy: { qc_mode: this.workItemQcMode() },
      ...this._briefPolicyField(),
    };

    this.saving.set(true);
    this.saveResult.set(null);

    const isExistingUser = this.selectedName() !== null && !this.isPackageDefault();
    const req$ = isExistingUser
      ? this.http.put<BlueprintContent>(`api/config/blueprints/${this.selectedName()}`, req)
      : this.http.post<BlueprintContent>('api/config/blueprints', req);

    req$.subscribe({
      next: (bp) => {
        this.saving.set(false);
        this.selectedName.set(bp.name);
        this.isPackageDefault.set(false);
        this.saveResult.set({ ok: true, message: 'Blueprint saved.' });
        this._refreshSidebar();
        this.router.navigate(['/blueprints', bp.name], { replaceUrl: true });
      },
      error: (err) => {
        this.saving.set(false);
        this.saveResult.set({ ok: false, message: err?.error?.detail ?? 'Save failed.' });
      },
    });
  }

  deleteBlueprint(): void {
    const name = this.selectedName();
    if (!name) return;
    this.deleting.set(true);
    this.http.delete(`api/config/blueprints/${name}`).subscribe({
      next: () => {
        this.deleting.set(false);
        this._refreshSidebar();
        this.newBlueprint();
      },
      error: (err) => {
        this.deleting.set(false);
        this.saveResult.set({ ok: false, message: err?.error?.detail ?? 'Delete failed.' });
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Agent editing
  // ---------------------------------------------------------------------------

  addAgent(): void {
    this.agents.update(list => [...list, {
      key: '', prompt: '', ragEnabled: false, commands: [], codeSamples: [],
    }]);
    this.importPaths.update(p => [...p, '']);
    this.importErrors.update(e => [...e, null]);
    this.importing.update(f => [...f, false]);
  }

  removeAgent(index: number): void {
    this.agents.update(list => list.filter((_, i) => i !== index));
    this.importPaths.update(p => p.filter((_, i) => i !== index));
    this.importErrors.update(e => e.filter((_, i) => i !== index));
    this.importing.update(f => f.filter((_, i) => i !== index));
  }

  addCommand(agentIndex: number): void {
    this.agents.update(list => {
      const updated = [...list];
      updated[agentIndex] = {
        ...updated[agentIndex],
        commands: [...updated[agentIndex].commands, { key: '', target_agent: '', description: '' }],
      };
      return updated;
    });
  }

  removeCommand(agentIndex: number, cmdIndex: number): void {
    this.agents.update(list => {
      const updated = [...list];
      updated[agentIndex] = {
        ...updated[agentIndex],
        commands: updated[agentIndex].commands.filter((_, i) => i !== cmdIndex),
      };
      return updated;
    });
  }

  addCodeSample(agentIndex: number): void {
    this.agents.update(list => {
      const updated = [...list];
      updated[agentIndex] = {
        ...updated[agentIndex],
        codeSamples: [...updated[agentIndex].codeSamples, ''],
      };
      return updated;
    });
  }

  removeCodeSample(agentIndex: number, sampleIndex: number): void {
    this.agents.update(list => {
      const updated = [...list];
      updated[agentIndex] = {
        ...updated[agentIndex],
        codeSamples: updated[agentIndex].codeSamples.filter((_, i) => i !== sampleIndex),
      };
      return updated;
    });
  }

  updateAgentField<K extends keyof AgentEntry>(agentIndex: number, field: K, value: AgentEntry[K]): void {
    this.agents.update(list => {
      const updated = [...list];
      updated[agentIndex] = { ...updated[agentIndex], [field]: value };
      return updated;
    });
  }

  updateCommandField<K extends keyof CommandEntry>(agentIndex: number, cmdIndex: number, field: K, value: CommandEntry[K]): void {
    this.agents.update(list => {
      const updated = [...list];
      const cmds = [...updated[agentIndex].commands];
      cmds[cmdIndex] = { ...cmds[cmdIndex], [field]: value };
      updated[agentIndex] = { ...updated[agentIndex], commands: cmds };
      return updated;
    });
  }

  updateCodeSample(agentIndex: number, sampleIndex: number, value: string): void {
    this.agents.update(list => {
      const updated = [...list];
      const samples = [...updated[agentIndex].codeSamples];
      samples[sampleIndex] = value;
      updated[agentIndex] = { ...updated[agentIndex], codeSamples: samples };
      return updated;
    });
  }

  updateImportPath(agentIndex: number, value: string): void {
    this.importPaths.update(p => { const n = [...p]; n[agentIndex] = value; return n; });
    this.importErrors.update(e => { const n = [...e]; n[agentIndex] = null; return n; });
  }

  importCodeSample(agentIndex: number): void {
    const path = (this.importPaths()[agentIndex] ?? '').trim();
    if (!path) return;

    this.importing.update(f => { const n = [...f]; n[agentIndex] = true; return n; });
    this.importErrors.update(e => { const n = [...e]; n[agentIndex] = null; return n; });

    this.http.post<{ filename: string; destination: string }>(
      'api/config/code-samples/import',
      { source_path: path },
    ).subscribe({
      next: (res) => {
        this.importing.update(f => { const n = [...f]; n[agentIndex] = false; return n; });
        this.importPaths.update(p => { const n = [...p]; n[agentIndex] = ''; return n; });
        this.addCodeSampleValue(agentIndex, res.filename);
      },
      error: (err) => {
        this.importing.update(f => { const n = [...f]; n[agentIndex] = false; return n; });
        this.importErrors.update(e => {
          const n = [...e];
          n[agentIndex] = err?.error?.detail ?? 'Import failed.';
          return n;
        });
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Code sample autocomplete
  // ---------------------------------------------------------------------------

  onSampleInput(ai: number, si: number, value: string): void {
    this.updateCodeSample(ai, si, value);
    this.sampleDropdown.set({ ai, si, query: value });
  }

  onSampleFocus(ai: number, si: number, value: string): void {
    if (this._blurTimer !== null) {
      clearTimeout(this._blurTimer);
      this._blurTimer = null;
    }
    this.sampleDropdown.set({ ai, si, query: value });
  }

  onSampleBlur(): void {
    this._blurTimer = setTimeout(() => {
      this.sampleDropdown.set(null);
      this._blurTimer = null;
    }, 150);
  }

  selectSuggestion(ai: number, si: number, filename: string): void {
    if (this._blurTimer !== null) {
      clearTimeout(this._blurTimer);
      this._blurTimer = null;
    }
    this.updateCodeSample(ai, si, filename);
    this.sampleDropdown.set(null);
  }

  isSampleDropdownVisible(ai: number, si: number): boolean {
    const dd = this.sampleDropdown();
    return dd !== null && dd.ai === ai && dd.si === si && this.filteredSuggestions().length > 0;
  }

  isBuiltin(filename: string): boolean {
    return this.availableCodeSamples().find(s => s.filename === filename)?.is_builtin ?? false;
  }

  agentKeys = computed(() => this.agents().map(a => a.key));

  goBack(): void {
    this.router.navigate(['/']);
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  addCodeSampleValue(agentIndex: number, value: string): void {
    this.agents.update(list => {
      const updated = [...list];
      updated[agentIndex] = {
        ...updated[agentIndex],
        codeSamples: [...updated[agentIndex].codeSamples, value],
      };
      return updated;
    });
  }

  private _resetImportState(count: number): void {
    this.importPaths.set(Array(count).fill(''));
    this.importErrors.set(Array(count).fill(null));
    this.importing.set(Array(count).fill(false));
  }

  private _isUserBlueprint(name: string): boolean {
    const bp = this.allBlueprints().find(b => b.name === name);
    return bp ? !bp.is_package_default : false;
  }

  private _toAgentEntries(agents: Record<string, AgentConfig>): AgentEntry[] {
    return Object.entries(agents).map(([key, a]) => ({
      key,
      prompt: a.prompt,
      ragEnabled: a.rag_enabled,
      commands: Object.entries(a.neighbors).map(([cmdKey, cmd]) => ({
        key: cmdKey,
        target_agent: cmd.target_agent,
        description: cmd.description,
      })),
      codeSamples: a.code_samples ?? [],
    }));
  }

  private _toAgentConfigMap(entries: AgentEntry[]): Record<string, AgentConfig> {
    const map: Record<string, AgentConfig> = {};
    for (const entry of entries) {
      const neighbors: Record<string, CommandConfig> = {};
      for (const cmd of entry.commands) {
        neighbors[cmd.key] = { target_agent: cmd.target_agent, description: cmd.description };
      }
      map[entry.key] = {
        prompt: entry.prompt,
        rag_enabled: entry.ragEnabled,
        neighbors,
        code_samples: entry.codeSamples.filter(s => s.trim()),
      };
    }
    return map;
  }

  /** `{ brief_policy }` when the blueprint has one (object or null), else `{}`
   *  so an absent policy stays absent rather than becoming null. */
  private _briefPolicyField(): { brief_policy?: BriefPolicyConfig | null } {
    const briefPolicy = this.briefPolicy();
    return briefPolicy === undefined ? {} : { brief_policy: briefPolicy };
  }

  private _serializeToJson(): string {
    return JSON.stringify({
      name: this.blueprintName(),
      global_policy: this.globalPolicy(),
      evaluator_agent: this.evaluatorAgent(),
      work_item_policy: { qc_mode: this.workItemQcMode() },
      ...this._briefPolicyField(),
      agents: this._toAgentConfigMap(this.agents()),
    }, null, 2);
  }

  private _applyJsonToForm(raw: string): string | null {
    try {
      const parsed = JSON.parse(raw);
      // Absent key = blueprint has no brief_policy; otherwise it must be an
      // object or null and is kept verbatim.
      const briefPolicy = parsed.brief_policy;
      if (briefPolicy !== undefined && briefPolicy !== null &&
          (typeof briefPolicy !== 'object' || Array.isArray(briefPolicy))) {
        return 'brief_policy must be a JSON object or null.';
      }
      this.briefPolicy.set(briefPolicy);
      if (parsed.name !== undefined) this.blueprintName.set(parsed.name);
      if (parsed.global_policy !== undefined) this.globalPolicy.set(parsed.global_policy);
      if (parsed.evaluator_agent !== undefined) this.evaluatorAgent.set(parsed.evaluator_agent);
      this.workItemQcMode.set(parsed.work_item_policy?.qc_mode ?? 'optional');
      if (parsed.agents && typeof parsed.agents === 'object') {
        const entries = this._toAgentEntries(parsed.agents);
        this.agents.set(entries);
        this._resetImportState(entries.length);
      }
      return null;
    } catch (e) {
      return 'Invalid JSON: ' + (e as Error).message;
    }
  }

  private _validate(agents: AgentEntry[], name: string, evaluatorAgent: string | null): string[] {
    const errors: string[] = [];
    if (!name.trim()) errors.push('Blueprint name is required.');
    if (name.includes('/') || name.includes('\\') || name.endsWith('.json')) {
      errors.push('Blueprint name must not contain path separators or .json extension.');
    }
    if (agents.length === 0) errors.push('At least one agent is required.');
    const keys = new Set<string>();
    for (const agent of agents) {
      if (!agent.key.trim()) { errors.push('All agent names must be non-empty.'); continue; }
      if (keys.has(agent.key)) errors.push(`Duplicate agent name: '${agent.key}'.`);
      keys.add(agent.key);
    }
    for (const agent of agents) {
      for (const cmd of agent.commands) {
        if (cmd.target_agent && !keys.has(cmd.target_agent)) {
          errors.push(`Agent '${agent.key}': command '${cmd.key}' references unknown agent '${cmd.target_agent}'.`);
        }
      }
    }
    if (evaluatorAgent && !keys.has(evaluatorAgent)) {
      errors.push(`Evaluator agent '${evaluatorAgent}' does not match any defined agent.`);
    }
    if (this.workItemQcMode() === 'required' && !evaluatorAgent) {
      errors.push('Required work-item QC needs an evaluator agent.');
    }
    return errors;
  }

  private _refreshSidebar(): void {
    this.configSvc.loadAll().subscribe(() => {
      this.allBlueprints.set(this.configSvc.blueprints());
    });
  }
}
