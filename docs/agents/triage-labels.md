# Triage label vocabulary

The five canonical triage roles and the label strings used in this repo:

| Role | Label string | Meaning |
|------|-------------|---------|
| Needs evaluation | `needs-triage` | A maintainer needs to review and classify this issue |
| Waiting on reporter | `needs-info` | Blocked — waiting for more information from the person who filed it |
| AFK-agent-ready | `ready-for-agent` | Fully specified; an autonomous agent can pick this up with no human context |
| Human-ready | `ready-for-human` | Fully specified; requires human implementation or judgment |
| Won't fix | `wontfix` | This will not be actioned; close after applying |

## State machine

```
[new issue] → needs-triage
  → needs-info          (if reporter context is missing)
    → needs-triage      (once reporter provides info)
  → ready-for-agent     (if fully specified and automatable)
  → ready-for-human     (if fully specified but needs a human)
  → wontfix             (if out of scope / duplicate / declined)
```

## Ensuring labels exist in the repo

Before applying a label for the first time, verify it exists:

```bash
gh label list
```

Create missing labels (pick a color that matches your convention):

```bash
gh label create "needs-triage"     --color "#e4e669"
gh label create "needs-info"       --color "#d93f0b"
gh label create "ready-for-agent"  --color "#0075ca"
gh label create "ready-for-human"  --color "#008672"
gh label create "wontfix"          --color "#ffffff"
```
