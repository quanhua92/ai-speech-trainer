"""Typer CLI entry point for ai-speech-shadowing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from ai_speech_shadowing.core.audio import TARGET_SAMPLE_RATE, AudioSample
from ai_speech_shadowing.core.feedback import (
    DEFAULT_WEIGHTS,
    evaluate,
    to_json,
    to_markdown,
    to_terminal,
)
from ai_speech_shadowing.core.fluency import compare_fluency
from ai_speech_shadowing.core.history import (
    DEFAULT_HISTORY_DIR,
    format_summary,
    list_reports,
    load_report,
    save_report,
)
from ai_speech_shadowing.core.phoneme import get_extractor
from ai_speech_shadowing.core.preprocess import preprocess
from ai_speech_shadowing.core.prosody import (
    DEFAULT_PITCH_CEILING,
    DEFAULT_PITCH_FLOOR,
    extract_pitch,
)
from ai_speech_shadowing.tts.generator import (
    ReferenceConfig,
    ReferenceManager,
    parse_sentence_list,
    slugify,
)

app = typer.Typer(
    name="ai-speech-shadowing",
    help="Local-first speech shadowing evaluation engine.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress non-warning logs.")
    ] = False,
) -> None:
    """ai-speech-shadowing — local-first speech evaluation."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command("version")
def version_cmd() -> None:
    """Print the installed package version."""
    from ai_speech_shadowing import __version__

    typer.echo(__version__)


@app.command("preprocess")
def preprocess_cmd(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input audio file (WAV/FLAC/OGG — anything soundfile reads).",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            help="Output WAV path. Default: <input>.preprocessed.wav.",
        ),
    ] = None,
    target_sr: Annotated[
        int, typer.Option("--target-sr", help="Target sample rate in Hz.")
    ] = TARGET_SAMPLE_RATE,
    trim_top_db: Annotated[
        int,
        typer.Option(
            "--trim-top-db",
            help="Top-dB threshold for silence trimming. Pass 0 to disable.",
        ),
    ] = 30,
    normalize: Annotated[
        str,
        typer.Option(
            "--normalize",
            help="Volume normalization: 'peak', 'rms', or 'none'.",
        ),
    ] = "peak",
) -> None:
    """Preprocess an audio file: mono → resample → trim → normalize."""
    sample = AudioSample.from_wav(input)
    result = preprocess(
        sample,
        target_sr=target_sr,
        trim_top_db=trim_top_db if trim_top_db > 0 else None,
        normalize=None if normalize.lower() == "none" else normalize,
    )
    out = output or input.with_suffix(".preprocessed.wav")
    result.to_wav(out)
    typer.echo(
        f"wrote {out}  ({result.duration:.3f}s, {result.sample_rate} Hz, {result.channels}ch)"
    )


@app.command("phoneme")
def phoneme_cmd(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input audio file to extract phonemes from.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Phoneme model key: 'slplab-l2' (default) or 'espeak'. "
            "Overridable via the PHONEME_MODEL env var.",
        ),
    ] = "slplab-l2",
    device: Annotated[
        str, typer.Option("--device", help="'auto', 'cpu', 'mps', or 'cuda'.")
    ] = "auto",
    no_preprocess: Annotated[
        bool,
        typer.Option(
            "--no-preprocess",
            help="Skip preprocessing (input must already be 16kHz mono).",
        ),
    ] = False,
) -> None:
    """Extract the IPA phoneme sequence from an audio file via Wav2Vec2-CTC."""
    from ai_speech_shadowing.core.phoneme import get_phoneme_model

    sample = AudioSample.from_wav(input)
    canonical = sample if no_preprocess else preprocess(sample)
    extractor = get_phoneme_model(key=model, device=device)
    result = extractor.extract(canonical)
    typer.echo(result.raw_text if result.raw_text else "(no phonemes detected)")


@app.command("prosody")
def prosody_cmd(
    input: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input audio file to analyse pitch/prosody.",
        ),
    ],
    pitch_floor: Annotated[
        float, typer.Option("--pitch-floor", help="Praat pitch floor in Hz.")
    ] = DEFAULT_PITCH_FLOOR,
    pitch_ceiling: Annotated[
        float, typer.Option("--pitch-ceiling", help="Praat pitch ceiling in Hz.")
    ] = DEFAULT_PITCH_CEILING,
    no_preprocess: Annotated[
        bool,
        typer.Option(
            "--no-preprocess",
            help="Skip preprocessing (downmix/trim/normalize).",
        ),
    ] = False,
) -> None:
    """Extract F0 pitch statistics (mean, range, voiced ratio, …) from audio."""
    sample = AudioSample.from_wav(input)
    canonical = sample if no_preprocess else preprocess(sample)
    stats = extract_pitch(canonical, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    if not stats.is_voiced:
        typer.echo("(no voiced frames detected)")
        return
    typer.echo(f"mean {stats.mean_hz:.1f} Hz | median {stats.median_hz:.1f} Hz")
    typer.echo(
        f"min {stats.min_hz:.1f} Hz | max {stats.max_hz:.1f} Hz | range {stats.range_hz:.1f} Hz"
    )
    typer.echo(f"std {stats.std_hz:.1f} Hz | voiced {stats.voiced_ratio * 100:.1f}%")


@app.command("fluency")
def fluency_cmd(
    reference: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Reference (native) audio file.",
        ),
    ],
    hypothesis: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="User (hypothesis) audio file to compare against the reference.",
        ),
    ],
    min_pause: Annotated[
        float,
        typer.Option("--min-pause", help="Minimum interior pause length (s) to flag."),
    ] = 0.25,
    no_preprocess: Annotated[
        bool,
        typer.Option("--no-preprocess", help="Skip preprocessing of both files."),
    ] = False,
) -> None:
    """Compare fluency/timing of a user recording vs. a reference (MFCC + DTW)."""
    prep = (lambda s: s) if no_preprocess else preprocess
    ref = prep(AudioSample.from_wav(reference))
    hyp = prep(AudioSample.from_wav(hypothesis))
    diff = compare_fluency(ref, hyp, min_pause_s=min_pause)

    typer.echo(
        f"DTW distance {diff.dtw.distance:.2f} | "
        f"normalized {diff.dtw.normalized_distance:.3f} (path {diff.dtw.path_length})"
    )
    typer.echo(f"score {diff.score * 100:.0f}/100 ({diff.grade})")
    typer.echo(
        f"pauses: ref {diff.reference_pauses.count} "
        f"({diff.reference_pauses.total_seconds:.2f}s) | "
        f"hyp {diff.hypothesis_pauses.count} "
        f"({diff.hypothesis_pauses.total_seconds:.2f}s)"
    )
    typer.echo(
        f"syllable rate: ref {diff.syllable_rate_reference:.2f}/s | "
        f"hyp {diff.syllable_rate_hypothesis:.2f}/s"
    )


def _parse_weights(raw: str) -> tuple[float, float, float]:
    parts = [float(x) for x in raw.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter("expected three comma-separated values, e.g. '0.4,0.3,0.3'")
    if sum(parts) == 0:
        raise typer.BadParameter("at least one scoring weight must be non-zero")
    return parts[0], parts[1], parts[2]


@app.command("evaluate")
def evaluate_cmd(
    reference: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Reference (native) audio file.",
        ),
    ],
    hypothesis: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="User (hypothesis) audio file to evaluate.",
        ),
    ],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'terminal', 'json', or 'markdown'.",
        ),
    ] = "terminal",
    weights: Annotated[
        str,
        typer.Option(
            "--weights",
            help="Composite weights (pron,into,flu), e.g. '0.4,0.3,0.3'.",
        ),
    ] = ",".join(str(w) for w in DEFAULT_WEIGHTS),
    no_preprocess: Annotated[
        bool, typer.Option("--no-preprocess", help="Skip preprocessing of both files.")
    ] = False,
    no_save: Annotated[
        bool,
        typer.Option("--no-save", help="Do not persist the report to the history directory."),
    ] = False,
    history_dir: Annotated[
        Path, typer.Option("--history-dir", help="Where to save the report.")
    ] = DEFAULT_HISTORY_DIR,
    reference_text: Annotated[
        str | None,
        typer.Option(
            "--reference-text",
            help="The native sentence (enables best-effort word-level highlighting).",
        ),
    ] = None,
) -> None:
    """Full evaluation: pronunciation + prosody + fluency → unified report.

    Loads the Wav2Vec2 phoneme model on first use (~1.2 GB download if cached).
    The report is saved to the history directory unless --no-save is given.
    """
    prep = (lambda s: s) if no_preprocess else preprocess
    ref = prep(AudioSample.from_wav(reference))
    hyp = prep(AudioSample.from_wav(hypothesis))
    typer.echo("Loading model & evaluating...", err=True)
    report = evaluate(ref, hyp, weights=_parse_weights(weights), reference_text=reference_text)

    if not no_save:
        path = save_report(report, history_dir=history_dir)
        typer.echo(f"saved {path}", err=True)

    rendered = {
        "terminal": to_terminal,
        "json": to_json,
        "markdown": to_markdown,
    }.get(fmt.lower())
    if rendered is None:
        raise typer.BadParameter(f"unknown format {fmt!r}; use terminal|json|markdown")
    typer.echo(rendered(report))


@app.command("generate-reference")
def generate_reference_cmd(
    text: Annotated[
        str | None,
        typer.Option("--text", "-t", help="Single sentence to synthesize."),
    ] = None,
    list_file: Annotated[
        Path | None,
        typer.Option(
            "--list",
            "-l",
            exists=True,
            dir_okay=False,
            readable=True,
            help="File of sentences (one per line; '#'-prefixed lines are comments).",
        ),
    ] = None,
    voice: Annotated[str, typer.Option("--voice", help="Kokoro voice name.")] = "af_heart",
    lang: Annotated[
        str, typer.Option("--lang", help="Kokoro single-letter language code (a=en-us).")
    ] = "a",
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="References base directory.")
    ] = Path("data/references"),
    force: Annotated[
        bool, typer.Option("--force", help="Regenerate even if a cached reference exists.")
    ] = False,
) -> None:
    """Generate a Kokoro TTS reference (or a batch) under data/references/<slug>/.

    Provide exactly one of --text (single) or --list (batch file). The first
    call downloads the Kokoro-82M model (~330 MB).
    """
    if text is None and list_file is None:
        raise typer.BadParameter("provide --text <sentence> or --list <file>")
    if text is not None and list_file is not None:
        raise typer.BadParameter("provide --text OR --list, not both")

    config = ReferenceConfig(base_dir=output_dir, default_voice=voice, default_lang=lang)
    mgr = ReferenceManager(config)

    if text is not None:
        out = mgr.generate(text, voice=voice, lang=lang, force=force)
        typer.echo(f"wrote {out}  (slug: {slugify(text)})")
    else:
        sentences = parse_sentence_list(list_file)
        if not sentences:
            raise typer.BadParameter(f"no sentences found in {list_file}")
        paths = mgr.generate_batch(sentences, voice=voice, lang=lang, force=force)
        typer.echo(f"generated {len(paths)} reference(s) under {output_dir}")
        for p in paths:
            typer.echo(f"  {p}")


@app.command("backfill-phonemes")
def backfill_phonemes_cmd(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="References base directory.")
    ] = Path("data/references"),
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Recompute phonemes even for references that already have them."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would change without writing any files."),
    ] = False,
) -> None:
    """Populate metadata.json[\"phonemes\"] for references that predate capture-at-gen.

    For every reference under ``--output-dir`` with a stored ``text`` field, run
    misaki G2P on the text and persist the normalized espeak tokens to the
    ``phonemes`` block — the same field that :meth:`ReferenceManager.generate`
    populates at synthesis time. References that already have the block are
    skipped unless ``--force`` is given.

    Idempotent: safe to re-run. Use this once after upgrading to the
    capture-at-synthesis change to migrate existing references without
    re-synthesizing their audio.
    """
    from ai_speech_shadowing.core.g2p import text_to_espeak_tokens

    config = ReferenceConfig(base_dir=output_dir)
    mgr = ReferenceManager(config)

    refs = mgr.list_references()
    if not refs:
        typer.echo(f"no references found under {output_dir}")
        return

    updated = skipped = failed = 0
    for ref in refs:
        slug = str(ref.get("slug", ""))
        text = str(ref.get("text", "")).strip()
        has_phonemes = isinstance(ref.get("phonemes"), dict) and bool(
            ref["phonemes"].get("tokens")  # type: ignore[union-attr]
        )
        if not text:
            typer.echo(f"  SKIP  {slug}  (no 'text' field)")
            failed += 1
            continue
        if has_phonemes and not force:
            typer.echo(f"  HAVE  {slug}  (use --force to recompute)")
            skipped += 1
            continue

        # text → G2P → normalize → tokenize (not just normalize; misaki must
        # run first to convert the reference text into phonemes).
        tokens = text_to_espeak_tokens(text)
        if not tokens:
            typer.echo(f"  FAIL  {slug}  (G2P produced no tokens for {text!r})")
            failed += 1
            continue

        if dry_run:
            typer.echo(f"  WOULD {slug}  ({len(tokens)} tokens)")
            updated += 1
            continue

        mgr.set_phonemes(slug, tokens)
        typer.echo(f"  WROTE {slug}  ({len(tokens)} tokens)")
        updated += 1

    summary = f"\n{updated} updated, {skipped} skipped, {failed} failed"
    if dry_run:
        summary += "  (dry-run — no files written)"
    typer.echo(summary)


@app.command("record")
def record_cmd(
    output: Annotated[
        Path,
        typer.Argument(dir_okay=False, help="Output WAV path."),
    ],
    duration: Annotated[float, typer.Option("--duration", "-d", help="Seconds to record.")] = 5.0,
    sample_rate: Annotated[
        int, typer.Option("--sample-rate", help="Recording sample rate (Hz).")
    ] = TARGET_SAMPLE_RATE,
) -> None:
    """Record user audio from the microphone and write a WAV file."""
    import sounddevice as sd
    import soundfile as sf

    typer.echo(f"Recording {duration:.1f}s at {sample_rate} Hz… (Ctrl-C to abort)", err=True)
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except KeyboardInterrupt as e:
        raise typer.Abort() from e
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio, sample_rate)
    typer.echo(f"wrote {output}  ({duration:.1f}s, {sample_rate} Hz, mono)")


_AUDIO_SUFFIXES: set[str] = {".wav", ".flac", ".ogg", ".aiff"}


@app.command("batch")
def batch_cmd(
    reference: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Reference (native) audio file to evaluate against.",
        ),
    ],
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            readable=True,
            help="Directory of user recordings to evaluate.",
        ),
    ],
    history_dir: Annotated[
        Path, typer.Option("--history-dir", help="Where to save reports.")
    ] = DEFAULT_HISTORY_DIR,
    no_preprocess: Annotated[
        bool, typer.Option("--no-preprocess", help="Skip preprocessing.")
    ] = False,
) -> None:
    """Evaluate every recording in a directory against one reference.

    Reports are saved to the history directory; a summary table is printed at
    the end. Loads the phoneme model once and reuses it.
    """
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    recordings = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in _AUDIO_SUFFIXES)
    if not recordings:
        raise typer.BadParameter(f"no audio files found in {input_dir}")

    prep = (lambda s: s) if no_preprocess else preprocess
    ref = prep(AudioSample.from_wav(reference))
    typer.echo(f"Loading model & evaluating {len(recordings)} recording(s)…", err=True)
    extractor = get_extractor()

    results: list[tuple[str, int, str]] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(recordings))
        for wav in recordings:
            hyp = prep(AudioSample.from_wav(wav))
            report = evaluate(ref, hyp, phoneme_extractor=extractor)
            save_report(report, history_dir=history_dir)
            results.append((wav.name, report.composite_score, report.composite_grade))
            progress.advance(task)

    typer.echo(f"Evaluated {len(results)} recording(s) (reports saved to {history_dir}):")
    for name, score, grade in sorted(results, key=lambda r: -r[1]):
        typer.echo(f"  {score:>3}/100  {grade:<10}  {name}")


@app.command("report")
def report_cmd(
    report_id: Annotated[
        str | None,
        typer.Argument(help="Report id; omit to list all saved reports."),
    ] = None,
    history_dir: Annotated[
        Path, typer.Option("--history-dir", help="History directory.")
    ] = DEFAULT_HISTORY_DIR,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="View format: 'summary' (default) or 'json'."),
    ] = "summary",
    list_only: Annotated[
        bool,
        typer.Option("--list", help="Always list, even if a report id is given."),
    ] = False,
) -> None:
    """List saved evaluation reports, or view one in detail."""
    if report_id is None or list_only:
        entries = list_reports(history_dir)
        if not entries:
            typer.echo("(no saved reports)")
            return
        for entry in entries:
            typer.echo(
                f"{entry.id}  {entry.composite_score:>3}/100  "
                f"{entry.composite_grade:<10}  {entry.created_at}"
            )
        return

    data = load_report(report_id, history_dir)
    if data is None:
        raise typer.BadParameter(f"no report with id {report_id!r} in {history_dir}")
    if fmt.lower() == "json":
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        typer.echo(format_summary(data))


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
    ssl_certfile: Annotated[
        Path | None,
        typer.Option(
            "--ssl-certfile",
            help="TLS cert (enables HTTPS — needed for the mic from non-localhost).",
        ),
    ] = None,
    ssl_keyfile: Annotated[
        Path | None, typer.Option("--ssl-keyfile", help="TLS private key.")
    ] = None,
) -> None:
    """Serve the REST API (FastAPI + uvicorn) at /api/v1."""
    import uvicorn

    scheme = "https" if ssl_certfile else "http"
    typer.echo(f"Serving API on {scheme}://{host}:{port}/api/v1  (docs at /docs)")
    uvicorn.run(
        "ai_speech_shadowing.api.app:app",
        host=host,
        port=port,
        reload=reload,
        ssl_certfile=str(ssl_certfile) if ssl_certfile else None,
        ssl_keyfile=str(ssl_keyfile) if ssl_keyfile else None,
    )
