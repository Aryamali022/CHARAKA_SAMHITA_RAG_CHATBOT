// The backend returns answers with citations like "[S1]" (already checked and
// normalised by src/answer.py). Before the answer is rendered as Markdown,
// each citation to a returned source becomes a link "#cite-S1", which the
// answer card draws as a clickable chip that opens that source.

const CITATION_RE = /\[([SN]\d+)\]/g;
export const CITE_PREFIX = "#cite-";

export function linkCitations(answer: string, labels: Iterable<string>): string {
  const known = new Set(labels);
  return answer.replace(CITATION_RE, (match, label: string) =>
    known.has(label) ? `[${label}](${CITE_PREFIX}${label})` : match,
  );
}

/** "S1" from "#cite-S1", or null for any other link. */
export function citedLabel(href: string | undefined): string | null {
  return href?.startsWith(CITE_PREFIX) ? href.slice(CITE_PREFIX.length) : null;
}
