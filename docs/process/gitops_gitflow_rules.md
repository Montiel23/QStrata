# QStrata GitOps / GitFlow Rules

- **Status:** Active
- **Date:** 2026-05-24
- **Branch:** feature/data-understanding
- **Scope:** All contributors, all slices, all branches

---

## Purpose

This document defines the mandatory Git hygiene rules for the QStrata project. Every slice must follow these rules without exception. Deviations require explicit human architect approval before they occur — not after.

---

## Mandatory Rules

### Rule 1 — Every slice ends with a commit or an explicit no-commit decision

Every slice must end with either a clean commit (or set of commits) covering all work produced, or an explicit documented statement in the checkpoint summary explaining why no commit is needed. A slice is not complete if it leaves uncommitted work without documented justification.

### Rule 2 — No slice begins with a dirty working tree unless documented

Before a new slice begins, the working tree must be clean. If it is not clean, the dirtiness must be intentional, documented in the checkpoint summary before the slice starts, and explicitly approved by the human architect. Undocumented dirt is never acceptable.

### Rule 3 — Commits must be small and logically grouped

Docs, configs, source code, reports, and other file categories must be committed separately unless there is a specific, explicitly stated reason to combine them. Never mix unrelated concerns in a single commit. Commit messages must describe what changed — not what the slice was called.

### Rule 4 — Never commit notebooks or Docker files

Files matching `*.ipynb`, `Dockerfile`, `docker-compose*`, or any variant must never be committed unless the slice specification explicitly authorises it. Stage these files only when the slice prompt says so.

### Rule 5 — Never create or commit `qcore/nas/__init__.py`

This repository uses Python namespace packages. `qcore/nas/__init__.py` must never be created or staged. If it appears in the working tree, it must be deleted immediately and the cause investigated before any commits are made.

### Rule 6 — Never push, merge, or switch branches without explicit human approval

No `git push`, `git merge`, `git rebase`, or `git checkout <other-branch>` may be executed without an explicit instruction in the slice prompt or a direct human approval message. When in doubt, stop and ask.

### Rule 7 — Run `git status --short` before each new slice begins

The first action at the start of every slice must be to run `git status --short` and confirm the working tree is clean. If it is not clean, stop immediately, document what is present, and request a Git hygiene checkpoint before proceeding.

### Rule 8 — Run `git status` and `git log` after each slice completes

At the end of every slice checkpoint summary, include the output of:
```
git status --short
git log --oneline -5
```
This confirms that all intended work was committed, nothing unexpected remains, and the log accurately reflects the slice's contributions.

### Rule 9 — Uncommitted changes at end of slice require a Git checkpoint

If any modified or untracked files remain after a slice completes its checkpoint, stop immediately. Do not begin the next slice. Request a dedicated Git hygiene checkpoint to commit, restore, or document each remaining item before proceeding.

---

## Summary Table

| Rule | Short form |
|---|---|
| 1 | Commit or document no-commit at end of every slice |
| 2 | Clean tree before every slice (or document why not) |
| 3 | Small, logically grouped commits — no category mixing |
| 4 | Never commit `*.ipynb` or Docker files without explicit authorisation |
| 5 | Never create or commit `qcore/nas/__init__.py` |
| 6 | Never push, merge, or branch-switch without explicit human approval |
| 7 | `git status --short` as first action of every slice |
| 8 | `git status` + `git log` in every slice checkpoint summary |
| 9 | Uncommitted changes → Git checkpoint before next slice |
