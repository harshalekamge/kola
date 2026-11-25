# Kola — Multimodal AI Assistant for Accessible Screen Navigation and Personalized Interaction

**Kola** is a team research project: an AI assistant that lets people control their computer by voice or text — "go to Watch Later", "play the first Shorts video" — so that users with motor or visual impairments can navigate apps like YouTube without a mouse. Kola understands what's on the screen, plans the steps, acts with the mouse and keyboard, and adapts to each user over time.

This repository is the team's shared workspace. Each research component lives in its own folder (or its own repository), and GitHub activity is posted to the team's Microsoft Teams channel.

---

## Table of contents

1. [System overview](#system-overview)
2. [Components](#components)
3. [Repository layout](#repository-layout)
4. [Getting started](#getting-started)
5. [Branching and workflow](#branching-and-workflow)
6. [Teams notifications (GitHub Actions)](#teams-notifications-github-actions)
7. [Adding a new component](#adding-a-new-component)

---

## System overview

```
 User (voice / text)
        │
        ▼
 Language module ── understands the request, produces a plan (JSON steps)
        │
        ▼
 Kola Vision ─── sees the screen (YOLO + OCR), finds the target, moves / clicks / types,
        │        checks the screen changed
        ▼
 Personalization ─ learns the user's habits and preferences to tailor suggestions and interaction
```

The language module and Kola Vision talk through a simple list of JSON steps, for example:

```json
[{ "action": "left_click", "target": { "type": "watch_later" } }]
```

The full step format is documented in [kola-vision/docs/INTERFACE.md](https://github.com/harshalekamge/kola-vision/blob/main/docs/INTERFACE.md).

## Components

| Component | Where | Status |
|---|---|---|
| **Kola Vision** — screen understanding and action execution (YOLO11 element detection, OCR, grounding, planner, executor, desktop demo app) | [`harshalekamge/kola-vision`](https://github.com/harshalekamge/kola-vision) | Working demo |
| **Personalization** — per-user adaptation | [`personalization/`](personalization/) | Started |
| **Language module** — intent understanding and planning | to be added | Planned |

## Repository layout

```
kola/
├── .github/
│   └── workflows/
│       └── teams-notifications.yml   # Posts pushes and pull requests to Microsoft Teams
└── personalization/
    └── README.md                     # Personalization module (in progress)
```

## Getting started

### Prerequisites

| Tool | Version | Check |
|---|---|---|
| Git | any | `git --version` |
| Python | 3.10 – 3.12 (for the Python components) | `python --version` |

### Clone

```bash
git clone https://github.com/harshalekamge/kola.git
cd kola
```

### Run Kola Vision (the working demo)

Kola Vision is in its own repository:

```bash
git clone https://github.com/harshalekamge/kola-vision.git
cd kola-vision
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python kola_demo.py
```

Full instructions, demo commands and troubleshooting: [kola-vision README](https://github.com/harshalekamge/kola-vision#readme).

## Branching and workflow

- `main` — stable, reviewed work
- `feature/<name>` — one branch per feature or experiment, e.g. `feature/personalization-profile`

```bash
git checkout -b feature/my-change
# ...work, commit...
git push -u origin feature/my-change
# open a pull request into main on GitHub
```

Pushes to `main` and `feature/**`, and pull requests (opened, reopened, ready for review, closed/merged), are announced in Teams automatically.

## Teams notifications (GitHub Actions)

`.github/workflows/teams-notifications.yml` sends a message card to a Microsoft Teams channel for every push and pull request event, with the author, branch, commit message or PR title, and a link.

To set it up (repository admin):

1. In Microsoft Teams, open the channel → **⋯ → Workflows** (or **Connectors**) → create an **incoming webhook** and copy its URL.
2. In GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
3. Name: `TEAMS_WEBHOOK_URL`, value: the webhook URL.

No other configuration is needed; the workflow uses only Python's standard library on `ubuntu-latest`.

## Adding a new component

1. Create a folder at the repository root (e.g. `language/`).
2. Add a `README.md` inside it covering purpose, setup, how to run and how it connects to the step interface.
3. Keep dependencies in that folder (`requirements.txt` or `package.json`) so components don't clash.
4. Work on a `feature/...` branch and open a pull request.

---

Kola research team
