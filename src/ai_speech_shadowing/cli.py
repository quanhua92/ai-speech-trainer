"""Typer CLI entry point for ai-speech-shadowing."""

from __future__ import annotations

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
        typer.Option("--model", help="HuggingFace Wav2Vec2 phoneme model id."),
    ] = "facebook/wav2vec2-lv-60-espeak-cv-ft",
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
    sample = AudioSample.from_wav(input)
    canonical = sample if no_preprocess else preprocess(sample)
    extractor = get_extractor(model_id=model, device=device)
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
) -> None:
    """Full evaluation: pronunciation + prosody + fluency → unified report.

    Loads the Wav2Vec2 phoneme model on first use (~1.2 GB download if cached).
    """
    prep = (lambda s: s) if no_preprocess else preprocess
    ref = prep(AudioSample.from_wav(reference))
    hyp = prep(AudioSample.from_wav(hypothesis))
    report = evaluate(ref, hyp, weights=_parse_weights(weights))

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
