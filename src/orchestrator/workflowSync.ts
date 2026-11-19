import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import {
  buildMultiAgentOrchestratorWorkflow,
  ORCHESTRATOR_WORKFLOW_ID,
  type OrchestratorCredentials,
  type ProviderCredentialRef,
} from './workflowBuilder.ts';

export interface WorkflowSyncResult {
  workflowId: string;
  workflowName: string;
  exportedPath: string;
  credentialsUsed: OrchestratorCredentials;
  projectId: string | null;
  imported: boolean;
  published: boolean;
  skippedExisting: boolean;
}

const FALLBACK_RIVER_CREDENTIAL: ProviderCredentialRef = {
  id: 'riverAiCredV4101',
  name: 'River AI',
};

const FALLBACK_OPENROUTER_CREDENTIAL: ProviderCredentialRef = {
  id: 'o3dspeUuaShSW6fK',
  name: 'OpenRouter account',
};

function resolveSqlitePath(userFolder: string): string {
  return path.join(userFolder, '.n8n', 'database.sqlite');
}

function findN8nBinary(cwd: string): string | null {
  const candidates = [
    path.resolve(cwd, '.n8n-runtime/node_modules/.bin/n8n'),
    path.resolve(cwd, 'node_modules/.bin/n8n'),
    '/usr/local/bin/n8n',
    '/usr/bin/n8n',
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

function inspectDatabaseContext(sqlitePath: string): {
  exists: boolean;
  workflowExists: boolean;
  credentials: OrchestratorCredentials;
  projectId: string | null;
} {
  if (!fs.existsSync(sqlitePath)) {
    return {
      exists: false,
      workflowExists: false,
      credentials: {
        riverAi: FALLBACK_RIVER_CREDENTIAL,
        openRouter: FALLBACK_OPENROUTER_CREDENTIAL,
      },
      projectId: null,
    };
  }

  const db = new DatabaseSync(sqlitePath, { readOnly: true });
  try {
    const existingWorkflow = db
      .prepare('SELECT id FROM workflow_entity WHERE id = ?')
      .get(ORCHESTRATOR_WORKFLOW_ID) as { id?: string } | undefined;

    const riverRow = db
      .prepare(
        "SELECT id, name FROM credentials_entity WHERE type = 'openAiApi' AND (id = 'riverAiCredV4101' OR name LIKE '%River%') ORDER BY createdAt ASC LIMIT 1",
      )
      .get() as { id?: string; name?: string } | undefined;

    const openRouterRow = db
      .prepare(
        "SELECT id, name FROM credentials_entity WHERE type = 'openRouterApi' ORDER BY createdAt ASC LIMIT 1",
      )
      .get() as { id?: string; name?: string } | undefined;

    const projectRow = db
      .prepare("SELECT id FROM project WHERE type = 'personal' ORDER BY createdAt ASC LIMIT 1")
      .get() as { id?: string } | undefined;

    return {
      exists: true,
      workflowExists: Boolean(existingWorkflow?.id),
      credentials: {
        riverAi:
          riverRow?.id && riverRow?.name
            ? { id: riverRow.id, name: riverRow.name }
            : FALLBACK_RIVER_CREDENTIAL,
        openRouter:
          openRouterRow?.id && openRouterRow?.name
            ? { id: openRouterRow.id, name: openRouterRow.name }
            : FALLBACK_OPENROUTER_CREDENTIAL,
      },
      projectId: projectRow?.id ?? null,
    };
  } finally {
    db.close();
  }
}

function ensureWorkflowOwnership(
  sqlitePath: string,
  workflowId: string,
  projectId: string,
): void {
  const db = new DatabaseSync(sqlitePath);
  try {
    const personalProjects = db
      .prepare("SELECT id FROM project WHERE type = 'personal'")
      .all() as Array<{ id: string }>;

    const targetIds = new Set<string>(
      personalProjects.map((p) => p.id).filter(Boolean),
    );
    if (projectId) {
      targetIds.add(projectId);
    }

    const now = new Date().toISOString().replace('T', ' ').slice(0, 23);
    for (const pid of targetIds) {
      const existing = db
        .prepare(
          'SELECT workflowId FROM shared_workflow WHERE workflowId = ? AND projectId = ?',
        )
        .get(workflowId, pid) as { workflowId?: string } | undefined;

      if (!existing) {
        db.prepare(
          `INSERT INTO shared_workflow (workflowId, projectId, role, createdAt, updatedAt)
           VALUES (?, ?, 'workflow:owner', ?, ?)`,
        ).run(workflowId, pid, now, now);
      }
    }
  } finally {
    db.close();
  }
}

export function syncOrchestratorWorkflow(options: {
  cwd: string;
  userFolder: string;
  overwriteExisting?: boolean;
}): WorkflowSyncResult {
  const sqlitePath = resolveSqlitePath(options.userFolder);
  const ctx = inspectDatabaseContext(sqlitePath);

  const workflow = buildMultiAgentOrchestratorWorkflow(ctx.credentials);
  const workflowsDir = path.resolve(options.cwd, 'workflows');
  fs.mkdirSync(workflowsDir, { recursive: true });
  const exportedPath = path.join(
    workflowsDir,
    'multiagent-cognitive-orchestrator.json',
  );
  fs.writeFileSync(exportedPath, `${JSON.stringify(workflow, null, 2)}\n`, 'utf8');

  if (!ctx.exists) {
    return {
      workflowId: workflow.id,
      workflowName: workflow.name,
      exportedPath,
      credentialsUsed: ctx.credentials,
      projectId: null,
      imported: false,
      published: false,
      skippedExisting: false,
    };
  }

  if (ctx.workflowExists && !options.overwriteExisting) {
    if (ctx.projectId) {
      ensureWorkflowOwnership(sqlitePath, workflow.id, ctx.projectId);
    }
    return {
      workflowId: workflow.id,
      workflowName: workflow.name,
      exportedPath,
      credentialsUsed: ctx.credentials,
      projectId: ctx.projectId,
      imported: false,
      published: false,
      skippedExisting: true,
    };
  }

  const n8nBin = findN8nBinary(options.cwd);
  if (!n8nBin) {
    throw new Error('n8n CLI binary not found; cannot import workflow.');
  }

  const cliEnv: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (typeof v === 'string') {
      cliEnv[k] = v;
    }
  }
  cliEnv.N8N_USER_FOLDER = options.userFolder;

  const importRes = spawnSync(
    n8nBin,
    ['import:workflow', `--input=${exportedPath}`],
    {
      cwd: options.cwd,
      env: cliEnv,
      encoding: 'utf8',
    },
  );
  if (importRes.status !== 0) {
    throw new Error(
      `n8n import:workflow failed (${importRes.status}): ${importRes.stderr || importRes.stdout}`,
    );
  }

  if (ctx.projectId) {
    ensureWorkflowOwnership(sqlitePath, workflow.id, ctx.projectId);
  }

  const publishRes = spawnSync(
    n8nBin,
    ['publish:workflow', `--id=${workflow.id}`],
    {
      cwd: options.cwd,
      env: cliEnv,
      encoding: 'utf8',
    },
  );
  const published = publishRes.status === 0;
  if (!published) {
    console.warn(
      `[n8n-orchestrator] publish:workflow returned ${publishRes.status}: ${publishRes.stderr || publishRes.stdout}`,
    );
  }

  return {
    workflowId: workflow.id,
    workflowName: workflow.name,
    exportedPath,
    credentialsUsed: ctx.credentials,
    projectId: ctx.projectId,
    imported: true,
    published,
    skippedExisting: false,
  };
}

const isDirectExecution =
  process.argv[1] &&
  path.resolve(process.argv[1]).endsWith(path.join('src', 'orchestrator', 'workflowSync.ts'));

if (isDirectExecution) {
  const cwd = process.cwd();
  const userFolder = path.resolve(cwd, process.env.N8N_USER_FOLDER || './.n8n-data');
  const overwrite = process.argv.includes('--overwrite');
  const result = syncOrchestratorWorkflow({
    cwd,
    userFolder,
    overwriteExisting: overwrite,
  });
  console.log(JSON.stringify(result, null, 2));
}
