# Issue tracker: GitHub Issues

Issues for this project live in **GitHub Issues** on the repo's GitHub remote.

## CLI

All reads and writes use the [`gh` CLI](https://cli.github.com/). Assume it is installed and authenticated.

## Common operations

### Create an issue

```bash
gh issue create --title "<title>" --body "<body>" --label "<label>"
```

### List open issues

```bash
gh issue list --state open
```

### View an issue

```bash
gh issue view <number>
```

### Edit an issue (labels, title, body)

```bash
gh issue edit <number> --add-label "<label>"
gh issue edit <number> --remove-label "<label>"
gh issue edit <number> --title "<new title>"
gh issue edit <number> --body "<new body>"
```

### Close an issue

```bash
gh issue close <number>
```

### Comment on an issue

```bash
gh issue comment <number> --body "<comment>"
```

## Conventions

- One issue per unit of work.
- Issue body uses Markdown. Include acceptance criteria as a checklist when relevant.
- Link issues to PRs with `Closes #<number>` in the PR description.
- Apply triage labels (see `triage-labels.md`) to communicate state.
