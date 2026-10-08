import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Message } from '../../../core/models/session.model';
import { MarkdownPipe } from '../../pipes/markdown.pipe';
import { TooltipDirective } from '../../directives/tooltip.directive';
import { ToolCall, ToolChipComponent, classifyToolCall } from '../tool-chip/tool-chip';

// Matches the server's `extract_python_code_blocks` fence pattern (bare and
// ```python fences). Executed code is rendered separately as a "code" chat
// item, so we strip it from the assistant message to avoid showing it twice.
const FENCED_CODE_RE = /```(?:python)?[ \t]*\n[\s\S]*?^[ \t]*```[ \t]*$/gm;

function stripFencedCodeBlocks(content: string): string {
  return content.replace(FENCED_CODE_RE, '');
}

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [CommonModule, MarkdownPipe, TooltipDirective, ToolChipComponent],
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
