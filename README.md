# ai-speech-shadowing

A local-first speech evaluation engine for language learning via the **shadowing technique**. Compare recorded user audio against native TTS reference clips and get multi-dimensional feedback on **pronunciation, prosody, and fluency** — entirely offline, with zero per-evaluation cost.

> **Status:** Pre-Alpha — scaffolding phase.

For the full architecture, roadmap, and API specification, see [`docs/README.md`](docs/README.md).

## Quick start

```bash
brew install espeak-ng                 # kokoro English fallback dependency
uv sync                                # create .venv and install deps
uv run python scripts/explore_kokoro.py --text "Hello world"
```

## Practice sentences

The repo ships with **26 pre-generated Kokoro references** (`data/references/`)
covering common workplace English — meetings, deadlines, code review, small
talk, and more. See [`data/workplace-20.txt`](data/workplace-20.txt) for the
full list. Pick one in the demo, record your attempt, and get instant feedback.

To regenerate or add your own:

```bash
uv run ai-speech-shadowing generate-reference --list data/workplace-20.txt
```

## Development

After cloning, install dependencies and activate the project-local Git hooks:

```bash
uv sync                               # install runtime + dev deps into .venv
git config core.hooksPath githooks    # use the project's hooks (ruff on commit)
```

Common tasks (see [`Justfile`](Justfile), run `just` to list):

```bash
just sync        # uv sync — install runtime + dev deps
just lint        # ruff check
just format      # ruff format
just test        # pytest
just verify      # lint + format-check + test (CI gate)
just explore "Hello world"   # run the Kokoro TTS explorer
```

Equivalent raw commands:

```bash
uv run ruff check .                   # lint
uv run ruff format .                  # format
uv run pytest                         # run the test suite
```

The `pre-commit` hook runs `ruff check` and `ruff format --check`. Bypass it
for a single commit with `git commit --no-verify`. See [`githooks/README.md`](githooks/README.md)
for details.

## License

MIT — see [`LICENSE`](LICENSE).
