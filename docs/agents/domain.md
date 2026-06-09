# Domain docs: consumer rules

## Layout

This repo uses a **single-context** layout:

| Artifact | Path |
|----------|------|
| Domain context | `CONTEXT.md` (repo root) |
| Architecture decision records | `docs/adr/` |

## How to read these files

### CONTEXT.md

Read `CONTEXT.md` at the start of any task that requires understanding the project's domain language, bounded contexts, or business rules. It is the canonical source for:

- What the system does and why
- Core domain terms and their precise meanings
- What is explicitly out of scope

If `CONTEXT.md` does not exist yet, ask the user to create it before proceeding with architecture-level tasks.

### docs/adr/

Each file in `docs/adr/` is an Architecture Decision Record. File naming convention: `NNNN-short-title.md` (e.g. `0001-use-postgres.md`).

Read relevant ADRs when:
- You are about to make a decision that may have been made before
- A user asks why something is structured a certain way
- You are diagnosing a surprising constraint in the codebase

An ADR with status `Superseded` should be read for context but not treated as current guidance — follow the ADR that superseded it.

### Writing new ADRs

When a significant architectural decision is made during a session, propose writing an ADR. Use this minimal structure:

```markdown
# NNNN — Title

**Status:** Accepted  
**Date:** YYYY-MM-DD

## Context

What situation prompted this decision.

## Decision

What was decided.

## Consequences

What becomes easier, harder, or different as a result.
```
