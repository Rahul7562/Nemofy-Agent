import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import type { Duplex } from 'node:stream';
import httpProxy from 'http-proxy';
import {
  appendAuthCookieHeader,
  AuthCookieBridge,
} from './src/server/authCookieBridge.ts';
import {
  isHttpsProxyRequest,
  resolveN8nRuntimeConfig,
  sanitizeProxyResponseHeaders,
  shouldProxyRequestToN8n,
} from './src/server/n8nConfig.ts';
import { N8nProcessManager } from './src/server/n8nProcessManager.ts';

const cwd = process.cwd();
const config = resolveN8nRuntimeConfig(process.env, cwd);
const manager = new N8nProcessManager(config, cwd);
const authCookieBridge = new AuthCookieBridge();

const proxy = httpProxy.createProxyServer({
  target: config.targetOrigin,
  ws: true,
  xfwd: true,
  changeOrigin: false,
  preserveHeaderKeyCase: true,
  proxyTimeout: 600_000,
  timeout: 600_000,
});

proxy.on('proxyReq', (proxyReq, req) => {
  const hostHeader = req.headers.host;
  if (hostHeader) {
    proxyReq.setHeader('x-forwarded-host', hostHeader);
  }
  if (isHttpsProxyRequest(req.headers, config.publicBaseUrl)) {
    proxyReq.setHeader('x-forwarded-proto', 'https');
  }

  const fallbackToken = authCookieBridge.resolveFallbackTokenForHttp(req.headers);
  if (fallbackToken) {
    proxyReq.setHeader(
      'cookie',
      appendAuthCookieHeader(req.headers.cookie, fallbackToken),
    );
  }
});

proxy.on('proxyReqWs', (proxyReq, req) => {
  const hostHeader = req.headers.host;
  if (hostHeader) {
    proxyReq.setHeader('x-forwarded-host', hostHeader);
  }
  if (isHttpsProxyRequest(req.headers, config.publicBaseUrl)) {
    proxyReq.setHeader('x-forwarded-proto', 'https');
  }

  const fallbackToken = authCookieBridge.resolveFallbackTokenForWebSocket(
    req.headers,
    req.socket?.remoteAddress,
  );
  if (fallbackToken) {
    proxyReq.setHeader(
      'cookie',
      appendAuthCookieHeader(req.headers.cookie, fallbackToken),
    );
  }
});

proxy.on('proxyRes', (proxyRes, req) => {
  authCookieBridge.recordResponseCookies(
    proxyRes.headers['set-cookie'],
    req.headers,
    req.socket?.remoteAddress,
  );

  const httpsRequest = isHttpsProxyRequest(req.headers, config.publicBaseUrl);
  const sanitized = sanitizeProxyResponseHeaders(proxyRes.headers, {
    httpsRequest,
    targetOrigin: config.targetOrigin,
  });

  for (const key of Object.keys(proxyRes.headers)) {
    if (!(key in sanitized)) {
      delete proxyRes.headers[key];
    }
  }
  for (const [key, value] of Object.entries(sanitized)) {
    if (value !== undefined) {
      proxyRes.headers[key] = value;
    }
  }
});

proxy.on('error', (err, _req, res) => {
  console.error(`[n8n-proxy] Upstream proxy error: ${err.message}`);
  if (res && 'writeHead' in res && !res.headersSent) {
    res.writeHead(503, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        error: 'n8n upstream temporarily unavailable',
        status: manager.getStatus(),
      }),
    );
  }
});

async function startServer(): Promise<void> {
  const app = express();
  app.disable('x-powered-by');

  app.get('/_n8n_proxy/status', (_req, res) => {
    res.setHeader('cache-control', 'no-store');
    res.json(manager.getStatus());
  });

  app.post('/_n8n_proxy/restart', async (_req, res) => {
    void manager.restart();
    res.status(202).json(manager.getStatus());
  });

  // Proxy all application traffic directly to n8n as soon as it is ready
  app.use((req, res, next) => {
    const requestUrl = req.originalUrl || req.url || '/';
    if (shouldProxyRequestToN8n(requestUrl, manager.getPhase())) {
      proxy.web(req, res, { target: config.targetOrigin });
      return;
    }
    next();
  });

  // While n8n is initializing (or on /_n8n_proxy/boot), serve the React boot screen
  const isProd = process.env.NODE_ENV === 'production';
  const distDir = path.resolve(cwd, 'dist');
  const distIndex = path.join(distDir, 'index.html');

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });

    app.use((req, res, next) => {
      if (req.url === '/_n8n_proxy/boot' || req.url.startsWith('/_n8n_proxy/boot?')) {
        req.url = req.url.replace('/_n8n_proxy/boot', '/_n8n_proxy/boot/');
      } else if (!req.url.startsWith('/_n8n_proxy/boot/')) {
        req.url = '/_n8n_proxy/boot/';
      }
      vite.middlewares(req, res, next);
    });
  } else if (fs.existsSync(distIndex)) {
    app.use('/_n8n_proxy/boot', express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(distIndex);
    });
  } else {
    app.get('*', (_req, res) => {
      res.status(503).send('n8n is starting up. Please refresh in a few seconds.');
    });
  }

  const server = http.createServer(app);
  server.requestTimeout = 600_000;

  server.on('upgrade', (req, socket: Duplex, head: Buffer) => {
    const requestUrl = req.url || '/';
    if (shouldProxyRequestToN8n(requestUrl, manager.getPhase())) {
      proxy.ws(req, socket, head, { target: config.targetOrigin });
      return;
    }
    socket.destroy();
  });

  server.listen(config.proxyPort, '0.0.0.0', () => {
    console.log(
      `[n8n-proxy] Devserver listening on http://0.0.0.0:${config.proxyPort} (bridging n8n port ${config.n8nPort})`,
    );
    void manager.start();
  });

  const handleShutdown = async (signal: string) => {
    console.log(`[n8n-proxy] Received ${signal}; shutting down...`);
    await manager.stop();
    server.close(() => {
      process.exit(0);
    });
  };

  process.once('SIGINT', () => {
    void handleShutdown('SIGINT');
  });
  process.once('SIGTERM', () => {
    void handleShutdown('SIGTERM');
  });
}

void startServer();
