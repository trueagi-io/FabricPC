# Rebasing Your Feature Branch

Rebase your feature branch onto `origin/main` to incorporate recent changes. Do this:

- Before opening a pull request
- During development when you need updates from `main`

## Why rebase?

Rebasing replays your commits on top of the latest `main`, as if you had started your work from that point. This:

- Ensures your branch includes the latest changes before merging
- Produces a linear, readable commit history
- Surfaces any conflicts *before* the merge, so you can resolve them in your branch

## How to rebase

### 0. Prerequisites (person doing the merge)
	
Coordinate with your branch partner(s) to verify their commits are already pushed to remote.

Your local feature branch is up to date with your remote feature branch.

### 1. Fetch the latest changes

```bash
git fetch origin
```

### 2. Rebase onto origin/main

From your feature branch:

```bash
git rebase origin/main
```

### 3. Resolve conflicts (if any)

If a commit conflicts with changes in `main`, git will pause and let you fix it:

```bash
git status                  # see which files have conflicts
# edit the conflicting files to resolve
git add <resolved-file>
git rebase --continue
```

Repeat until all commits are replayed. To abort and start over:

```bash
git rebase --abort
```

### 4. Push your rebased branch

Rebasing rewrites commit history, so you need to force-push:

```bash
git push --force-with-lease
```

## Coordinating with your branch partner

If two developers are working on the same branch, coordinate before rebasing:

1. Let your partner know you're about to rebase
2. Ensure you've both pushed any local commits
3. One person rebases and force-pushes
4. The other person resets their local branch:

```bash
git fetch origin
git reset --hard origin/<branch-name>
```

This replaces their local branch with the rebased version. Any unpushed work would be lost, which is why step 2 matters.

## Quick reference

| Command | Purpose |
|---------|---------|
| `git fetch origin` | Get latest remote state |
| `git rebase origin/main` | Rebase current branch onto main |
| `git rebase --continue` | Continue after resolving conflicts |
| `git rebase --abort` | Cancel and restore original state |
| `git push --force-with-lease` | Push rewritten history safely |
| `git reset --hard origin/<branch>` | Sync local branch after partner rebases |
