import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Message } from '../../../core/models/session.model';
import { MarkdownPipe } from '../../pipes/markdown.pipe';
import { TooltipDirective } from '../../directives/tooltip.directive';
import { IconComponent } from '../icon/icon';

// Matches the server's `extract_python_code_blocks` fence pattern (bare and
// ```python fences). Executed code is rendered separately as a "code" chat
// item, so we strip it from the assistant message to avoid showing it twice.
const FENCED_CODE_RE = /```(?:python)?[ \t]*\n[\s\S]*?^[ \t]*```[ \t]*$/gm;

function stripFencedCodeBlocks(content: string): string {
  return content.replace(FENCED_CODE_RE, '');
}

interface ToolCall {
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

function classifyToolCall(content: string): ToolCall | null {
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

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [CommonModule, MarkdownPipe, TooltipDirective, IconComponent],
  templateUrl: './message-bubble.html',
  styleUrl: './message-bubble.scss',
})
export class MessageBubbleComponent {
  @Input() message!: Message;
  @Input() streaming = false;

  get displayContent(): string {
    if (this.streaming || this.message.role !== 'assistant') {
      return this.message.content;
    }
    return stripFencedCodeBlocks(this.message.content);
  }

  get toolCall(): ToolCall | null {
    if (this.streaming || this.message.role !== 'assistant') {
      return null;
    }
    return classifyToolCall(this.message.content);
  }

  get visible(): boolean {
    if (this.streaming || this.message.role !== 'assistant') {
      return true;
    }
    return this.displayContent.trim().length > 0;
  }
}
