/**
 * Pure text helpers for the collapsed code card header. No Angular.
 */

const SUMMARY_MAX_CHARS = 120;

function truncate(text: string): string {
  return text.length > SUMMARY_MAX_CHARS ? text.slice(0, SUMMARY_MAX_CHARS - 1) + '…' : text;
}

/**
 * One line that says what the code does: its first `#` comment with words in
 * it (not a `#!` line, not a `# ---` or `# %%` separator), otherwise its
 * first non-empty line. Empty source gives ''.
 */
export function codeSummary(source: string): string {
  const lines = source.split('\n').map(line => line.trim()).filter(line => line.length > 0);
  for (const line of lines) {
    if (!line.startsWith('#') || line.startsWith('#!')) continue;
    // Drop the '#' and decoration such as '# --- Load ---' or '# === QC ==='.
    const text = line.replace(/^#+/, '').replace(/^[-=*~\s]+|[-=*~\s]+$/g, '');
    if (/[A-Za-z0-9]/.test(text)) return truncate(text);
  }
  return lines.length ? truncate(lines[0]) : '';
}

/** Number of lines in the source, ignoring trailing newlines. */
export function codeLineCount(source: string): number {
  const body = source.replace(/\n+$/, '');
  return body.length ? body.split('\n').length : 0;
}

/** The last non-empty line of stderr (usually the exception), or null. */
export function lastStderrLine(stderr: string | undefined): string | null {
  if (!stderr) return null;
  const lines = stderr.split('\n').map(line => line.trim()).filter(line => line.length > 0);
  return lines.length ? lines[lines.length - 1] : null;
}
