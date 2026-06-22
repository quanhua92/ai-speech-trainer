# Long passage — reference vs. learner attempt

Source text for `long-report.{terminal.txt,json,md}`.

## Reference (native target)

> An LLM, or large **language model**, is a powerful computer program. It acts like a super smart assistant that understands human language.
>
> To learn, it reads massive **amounts** of text from books, articles, and websites. By looking at all these words, it learns how humans speak and write. It becomes an expert at guessing which word should **come** next in a sentence.
>
> Because of this training, it can do many amazing tasks. It can answer your **hard** questions, translate languages, and write creative **stories**. It can even code programs and chat **with** you just like a real person.
>
> Would you like to know how it learns or see examples of what it can do?

## Attempt (learner — 6 realistic mistakes, same voice)

> An LLM, or large **language motto**, is a powerful computer program. It acts like a super smart assistant that understands human language.
>
> To learn, it reads massive **amount** of text from books, articles, and websites. By looking at all these words, it learns how humans speak and write. It becomes an expert at guessing which word should **go** next in a sentence.
>
> Because of this training, it can do many amazing tasks. It can answer your **heard** questions, translate languages, and write creative **story**. It can even code programs and chat **to** you just like a real person.
>
> Would you like to know how it learns or see examples of what it can do?

Six scattered, realistic L2-learner errors: `model→motto`, `amounts→amount`, `come→go`, `hard→heard`, `stories→story`, `with→to`.

## Notes

- Both clips are real Kokoro (`af_heart`) synthesis at 24 kHz, ~40 s each.
- Composite **90/100** (PER 0.069 → pronunciation 93; intonation 95; fluency 82). The
  attempt is mostly correct, so the composite stays high — the exact phoneme strip
  pinpoints all 29 changed sounds.
- **Scoring is exact and deterministic**: evaluating the reference against *itself*
  yields PER = 0.000 (0 errors) — i.e. there is no random recognizer noise on long
  audio. The 29 phoneme diffs are real.
- **Word highlight flags 14 words, not 6.** That's correct, not noise: changing a word
  shifts the pronunciation of its neighbours (coarticulation, rhythm), and the engine
  detects those too. The 6 edited words plus ~8 affected neighbours all genuinely sound
  different from the native reference. Word-level alignment is per-sentence so its
  accuracy holds up on long passages.

## Reproduce

```bash
# generate both clips + the report (uses the scripts in /tmp during dev; the
# wavs live at data/recordings/long-{ref,attempt}.wav)
ai-speech-shadowing evaluate \
  data/recordings/long-ref.wav data/recordings/long-attempt.wav --no-save \
  --format json   # → examples/long-report.json
```
