# Nemofy Agent (Vercel-Ready Web Interface)

Standalone React + Vite + Tailwind CSS web workspace for the pure-Python `Agent-Core`, designed in sync with **https://nemofy.tech/** (full Light and Dark mode).

## Features
- **Protected Access Gate**: Requires administrator sign-in before accessing Chat, Filesystem, or Settings.
- **Chat**: Multi-session autonomous agent chat with execution plan viewer, sub-agent reports, active skill tracking, and Human-in-the-Loop approval controls.
- **Filesystem**: Live `/workspace` tree explorer, search, file editor, folder creation, and deletion.
- **Settings**:
  - **LLM Config**: Provider, Model ID, Base URL, Temperature, Max Tokens, and Reasoning Effort for both **Main Agent (Brain)** and **Sub-Agent (Worker)**, plus live connectivity probe.
  - **Change Password**: Update administrator credentials with immediate session rotation.
  - **Skills & MCPs**: Manage `/Skills/*.md` playbooks and `/MCPs/servers.json` endpoints.
  - **Appearance & Runtime**: Light/Dark theme switch and optional remote `Agent-Core` backend URL.

## Deploying to Vercel
1. Import this repository in Vercel.
2. Set **Root Directory** to `Nemofy Agent`.
3. Framework Preset: **Vite** (`npm run build`, output directory `dist`).
4. Optional: Set environment variable `VITE_AGENT_API_URL` to your hosted `server.ts` / `Agent-Core` gateway URL (or configure it inside **Settings → Appearance & Runtime**).
