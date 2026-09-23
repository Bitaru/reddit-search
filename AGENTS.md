# Agent instructions

Repository skills are stored in `.agents/skills/` using the Agent Skills format.
Read the matching skill before editing configuration or running a pipeline stage.

## Available skills

- [reddit-search pipeline](.agents/skills/reddit-search-pipeline/SKILL.md): use for source registration, ingestion, retrieval, blinded review, labels, evaluation, provenance, or pipeline debugging.
- [scenario authoring](.agents/skills/reddit-search-scenario-authoring/SKILL.md): use when creating or revising scenarios, lexical queries, semantic queries, or claim mappings.

If a task covers both areas, read the scenario-authoring skill before drafting the configuration, then use the pipeline skill to validate and run it.

The files under `.agents/skills/` are canonical. `.claude/skills/` contains discovery shims for Claude Code and must point back to the canonical skills instead of duplicating them.
