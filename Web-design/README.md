# Nemofy Web-Design System

Design tokens, typography, and brand assets synchronized with **https://nemofy.tech/** for the **Nemofy Agent** web application (`/Nemofy Agent`).

## Typography
- **Display & Headings**: `Outfit` (weights 500, 600, 700)
- **Body & Controls**: `Plus Jakarta Sans` (weights 400, 500, 600, 700)
- **Code & Telemetry**: `JetBrains Mono` (weights 400, 500)

## Color System
| Token | Light Mode | Dark Mode | Role |
| :--- | :--- | :--- | :--- |
| `--paper` | `#ffffff` | `#0b0f19` | Primary workspace canvas |
| `--paper-raised` | `#ffffff` | `#131926` | Elevated panels & cards |
| `--paper-sunk` | `#f4f7fa` | `#182032` | Sidebar & sunken surfaces |
| `--ink` | `#161c2d` | `#f8fafc` | Primary foreground text |
| `--ink-soft` | `#5a6474` | `#94a3b8` | Secondary labels & prose |
| `--signal` | `#473bf0` | `#635bff` | Primary interactive brand accent |
| `--rule` | `#e7e9ed` | `rgba(255,255,255,0.09)` | 1px structural hairline borders |
| `--jade` | `#10b981` | `#34d399` | Nominal / success state |
| `--honey` | `#f59e0b` | `#fbbf24` | Pending approval / warning |
| `--flare` | `#f64b4b` | `#f87171` | Destructive / error state |

## Layout Architecture
- **Protected Entry Gate**: Centered Nemofy sign-in panel before any agentic workspace or API data is accessible.
- **Persistent Left Sidebar (`260px`)**: Brand header + Primary navigation (**Chat**, **Filesystem**, **Settings**) + Light/Dark switch + User sign-out footer.
- **Single-Elevation Discipline**: Flat surfaces separated by `1px` `--rule` borders and clean whitespace.
