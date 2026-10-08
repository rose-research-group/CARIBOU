import { Component, input } from '@angular/core';
import { IconComponent } from '../icon/icon';

export interface ToolCall {
  label: string;
  command: string;
}

// Agent tool commands are emitted as bare, single-line commands. We surface
// them as a distinct "tool call" chip instead of plain prose so it is obvious
// to the user that the agent invoked a tool rather than wrote text.
const TOOL_CALL_PATTERNS: { re: RegExp; label: string }[] = [
  // Optional trailing numeric id matches legacy persisted delegation messages.
  { re: /^delegate_to_[A-Za-z0-9_]+(?: \d+)?$/, label: 'Delegation' },
  { re: /^query_rag_/, label: 'RAG query' },
  { re: /^open_work_item\b/, label: 'Open work item' },
  { re: /^close_work_item\b/, label: 'Close work item' },
  { re: /^list_work_items$/, label: 'List work items' },
  { re: /^read_work_item\b/, label: 'Read work item' },
  { re: /^end_session$/, label: 'End session' },
];

/** The tool call a message consists of, or null for ordinary prose. */
export function classifyToolCall(content: string): ToolCall | null {
  if (!content || content.includes('\n') || content.includes('`')) {
    return null;
  }
  const trimmed = content.trim();
  if (!trimmed) {
    return null;
  }
  for (const { re, label } of TOOL_CALL_PATTERNS) {
    if (re.test(trimmed)) {
      return { label, command: trimmed };
    }
  }
  return null;
}

/**
 * The "tool call" chip. An attribute component (`<div appToolChip>`) so the
 * chip's element is the host itself.
 */
@Component({
  selector: '[appToolChip]',
  standalone: true,
  imports: [IconComponent],
  template: `
    <span class="tool-call-badge">
      <app-icon name="zap" [size]="12" aria-hidden="true" />
      Tool call
    </span>
    <code class="tool-call-command">{{ call().command }}</code>
  `,
  styleUrl: './tool-chip.scss',
  host: {
    class: 'tool-call',
    '[attr.title]': 'call().label',
  },
})
export class ToolChipComponent {
  readonly call = input.required<ToolCall>();
}
