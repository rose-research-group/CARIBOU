import { Artifact } from '../models/session.model';

/**
 * Collapse artifacts to one entry per `path`, keeping the LAST occurrence
 * (the newest write of an overwritten file) at the position of the first.
 * `path` is the artifact's identity; an artifact without one is a contract
 * violation and throws rather than being guessed from `filename`.
 */
export function dedupeArtifactsByPath(artifacts: Artifact[]): Artifact[] {
  const byPath = new Map<string, Artifact>();
  for (const artifact of artifacts) {
    if (typeof artifact.path !== 'string' || typeof artifact.mtime_ns !== 'number') {
      throw new Error(`Artifact ${artifact.id} (${artifact.filename}) is missing path/mtime_ns`);
    }
    byPath.set(artifact.path, artifact);
  }
  return [...byPath.values()];
}

/** Absolute download URL of an artifact's file. */
export function artifactDownloadUrl(artifact: Artifact): string {
  const base = document.baseURI.replace(/\/$/, '');
  return `${base}/api/sessions/${artifact.session_id}/artifacts/${artifact.id}/download`;
}

/** Preview src; mtime_ns changes when the same path is overwritten, so the
 *  browser does not keep showing the cached image of the old file. */
export function artifactPreviewUrl(artifact: Artifact): string {
  return `${artifactDownloadUrl(artifact)}?v=${artifact.mtime_ns}`;
}
