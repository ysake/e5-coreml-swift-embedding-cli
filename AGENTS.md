# AGENTS.md

This repository is intended for agent-assisted implementation.

## Start here

Read these files before making changes:

1. `README.md`
2. `docs/agent-handoff.md`
3. `docs/implementation-plan.md`
4. `docs/model-conversion-notes.md`

For Japanese instructions, read:

1. `docs/agent-handoff.ja.md`
2. `docs/implementation-plan.ja.md`
3. `docs/model-conversion-notes.ja.md`

## Development policy

- Keep the first milestone small.
- Prefer a working SwiftPM CLI before adding iOS or visionOS samples.
- Keep Core ML inference behind a protocol.
- Keep tokenizer and model assets clearly separated.
- Add tests for pure Swift logic first.
- Do not commit large generated model artifacts unless the repository decides to use Git LFS.

## Commands

When implementation exists, run:

```bash
swift build
swift test
swift run e5-embed "テスト"
```
