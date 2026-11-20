import type { WorkspaceNode } from '../types/agent.ts';

export function formatBytes(bytes?: number): string {
  if (bytes === undefined || bytes < 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function normalizeWorkspacePath(input: string): string {
  const trimmed = input
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\/workspace\/?/, '')
    .replace(/^\/+/, '')
    .replace(/\/+$/, '');
  return trimmed;
}

export function isValidWorkspacePath(input: string): boolean {
  const clean = normalizeWorkspacePath(input);
  if (!clean) return false;
  if (clean.includes('\0')) return false;
  const segments = clean.split('/');
  return segments.every((seg) => seg.length > 0 && seg !== '.' && seg !== '..');
}

export function filterWorkspaceTree(nodes: WorkspaceNode[], query: string): WorkspaceNode[] {
  const q = query.trim().toLowerCase();
  if (!q) return nodes;
  const out: WorkspaceNode[] = [];
  for (const node of nodes) {
    const nameMatches =
      node.name.toLowerCase().includes(q) || node.path.toLowerCase().includes(q);
    if (node.type === 'folder') {
      const filteredChildren = filterWorkspaceTree(node.children || [], q);
      if (nameMatches || filteredChildren.length > 0) {
        out.push({
          ...node,
          children: nameMatches ? node.children || [] : filteredChildren,
        });
      }
    } else if (nameMatches) {
      out.push(node);
    }
  }
  return out;
}

export function detectFileLanguage(filePath: string): string {
  const ext = filePath.split('.').pop()?.toLowerCase() || '';
  const map: Record<string, string> = {
    py: 'Python',
    ts: 'TypeScript',
    tsx: 'TSX',
    js: 'JavaScript',
    jsx: 'JSX',
    json: 'JSON',
    md: 'Markdown',
    css: 'CSS',
    html: 'HTML',
    yaml: 'YAML',
    yml: 'YAML',
    sh: 'Shell',
    sql: 'SQL',
    csv: 'CSV',
    txt: 'Plain Text',
  };
  return map[ext] || 'Text';
}

export function countWorkspaceItems(nodes: WorkspaceNode[]): { files: number; folders: number } {
  let files = 0;
  let folders = 0;
  for (const node of nodes) {
    if (node.type === 'folder') {
      folders += 1;
      const sub = countWorkspaceItems(node.children || []);
      files += sub.files;
      folders += sub.folders;
    } else {
      files += 1;
    }
  }
  return { files, folders };
}
