import { Injectable, effect, signal } from '@angular/core';

const KEY = 'caribou:prefs:v1';

export interface UserPreferences {
  soundOnComplete: boolean;
  tabTitleNotifications: boolean;
  toastNotifications: boolean;
  slowCodeThresholdMs: number;
  developerMode: boolean;
}

const DEFAULTS: UserPreferences = {
  soundOnComplete: false,
  tabTitleNotifications: true,
  toastNotifications: true,
  slowCodeThresholdMs: 5000,
  developerMode: false,
};

@Injectable({ providedIn: 'root' })
export class PreferencesService {
  readonly prefs = signal<UserPreferences>(this.load());

  constructor() {
    effect(() => {
      const value = this.prefs();
      localStorage.setItem(KEY, JSON.stringify(value));
    });
  }

  patch(partial: Partial<UserPreferences>): void {
    this.prefs.update(p => ({ ...p, ...partial }));
  }

  private load(): UserPreferences {
    try {
      const raw = localStorage.getItem(KEY);
      if (!raw) return { ...DEFAULTS };
      const parsed = JSON.parse(raw) as Partial<UserPreferences>;
      return { ...DEFAULTS, ...parsed };
    } catch {
      return { ...DEFAULTS };
    }
  }
}
