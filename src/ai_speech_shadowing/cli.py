"""Typer CLI entry point for ai-speech-shadowing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ai_speech_shadowing.core.audio import TARGET_SAMPLE_RATE, AudioSample
from ai_speech_shadowing.core.phoneme import get_extractor
from ai_speech_shadowing.core.preprocess import preprocess

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
