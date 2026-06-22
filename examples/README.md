# Examples — sample evaluation reports

Real end-to-end evaluations so you can see exactly what the engine produces,
without running anything. Two scenarios:

| Example | Length | Shows | Files |
| --- | --- | --- | --- |
| **Fox** (short) | 9 words | full report **incl. word-level highlight** (exact phoneme diff too) | `report.*` |
| **LLM passage** (long) | ~75 words / ~40 s | report at scale — scores + feedback + the exact phoneme strip | `long-report.*`, `long-passage.md` |

> The fox example is the best showcase of the "where you're wrong" feature:
> both a **word highlight** (which words) and the **phoneme strip** (which sounds).
> The long example omits the word-level view — its best-effort alignment drifts
> over a ~430-phoneme sequence; the phoneme strip and scores stay exact at any length.

---

## Example 1 — Fox sentence (short, with word highlight)

A real end-to-end evaluation so you can see exactly what the engine produces,
without running anything. Generated on the bundled default reference.

## The scenario

| | Text | Voice |
| --- | --- | --- |
| **Reference (native)** | "The quick brown fox **jumps** over **the** lazy dog." | `af_heart` |
| **User attempt**       | "The quick brown fox **jumped** over **a** lazy dog." | `af_heart` |

Two small word changes (`jumps → jumped`, `the → a`) — exactly the kind of
mistake a learner might make. The engine is run on the actual Kokoro-synthesized
audio of both sentences (same voice, so intonation matches by construction;
only the changed sounds and their rhythm differ).

## How it was generated

```bash
# reference is already shipped under data/references/the-quick-brown-fox-jumps-over-the-lazy-dog/
ai-speech-shadowing generate-reference --text "The quick brown fox jumped over a lazy dog."
cp data/references/the-quick-brown-fox-jumped-over-a-lazy-dog/audio/kokoro-en-us/ref.wav \
   data/recordings/attempt-jumped-lazy.wav

REF=data/references/the-quick-brown-fox-jumps-over-the-lazy-dog/audio/kokoro-en-us/ref.wav
ATT=data/recordings/attempt-jumped-lazy.wav
ai-speech-shadowing evaluate "$REF" "$ATT" --format terminal    # examples/report.terminal.txt
ai-speech-shadowing evaluate "$REF" "$ATT" --format json        # examples/report.json
ai-speech-shadowing evaluate "$REF" "$ATT" --format markdown    # examples/report.md
```

## The report (terminal)

```
AI Speech Shadowing — Report
────────────────────────────────────────────────────
Pronunciation (PER):    90  🟢 good
Intonation (Pitch):    100  🟢 good
Fluency (DTW):          69  🟡 fair
────────────────────────────────────────────────────
Composite Score:       87/100  🟢 good
────────────────────────────────────────────────────
Feedback:
  • Your rhythm diverges from the reference; shadow the native pacing.
Words (best-effort):
  The quick brown fox [jumps] over [the] lazy dog
```

Two complementary "where you're wrong" views ship together:

- **Word highlight** (best-effort) — the reference sentence with the wrong
  *words* marked (`[jumps]`, `[the]`). Tells the learner **which** word.
- **Phoneme strip** (exact) — the `/s/→/t/`, `/ð/→/ɹ/` chips. Tells them **how**.

## Reading it

- **Pronunciation 90** — Phoneme Error Rate `0.097`. The diff caught both edits
  precisely: `jumps→jumped` shows as `/s/ → /t/`, and `the→a` as
  `/ð/ → /ɹ/` + `/ə/ → /ɐ/`. 3 substitutions out of 31 phonemes.
- **Intonation 100** — pitch-range ratio `0.998`; the same voice was used, so
  the contour matches. Not flagged as monotone.
- **Fluency 69** — normalized DTW `0.157`; the word changes shift the rhythm
  enough to drop this pillar to "fair" and trigger the pacing feedback.
- **Composite 87** — weighted `0.4·90 + 0.3·100 + 0.3·69 = 86.7 ≈ 87`.

## Files

| File | Format | Audience |
| --- | --- | --- |
| [`report.terminal.txt`](report.terminal.txt) | colour-coded terminal | humans (CLI) |
| [`report.md`](report.md) | Markdown table + feedback | docs / PR comments |
| [`report.json`](report.json) | full structured payload | programs (the API returns this shape) |

> The attempt audio lives at `data/recordings/attempt-jumped-lazy.wav`
> (gitignored — regenerable via the commands above).
