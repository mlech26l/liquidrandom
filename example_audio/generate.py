"""CLI entrypoint for synthetic audio-to-tool-calling dataset generation."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from llm import create_client
from conversation import generate_sample

app = typer.Typer()
console = Console()

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("persona", pa.string()),
        pa.field("tools_json", pa.string()),
        pa.field("conversation", pa.string()),
        pa.field("num_turns", pa.int32()),
        pa.field("num_tool_calls", pa.int32()),
    ]
)

AUDIO_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("persona", pa.string()),
        pa.field("tools_json", pa.string()),
        pa.field("conversation", pa.string()),
        pa.field("num_turns", pa.int32()),
        pa.field("num_tool_calls", pa.int32()),
        pa.field("voice", pa.string()),
        pa.field("user_audio", pa.list_(pa.binary())),
    ]
)


async def _generate_all(
    num_samples: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    client = create_client()
    semaphore = asyncio.Semaphore(batch_size)
    results: list[dict] = []

    async def generate_with_semaphore(sample_id: int) -> dict | None:
        async with semaphore:
            return await generate_sample(client, sample_id)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating conversations...", total=num_samples)

        tasks = [
            asyncio.create_task(generate_with_semaphore(i))
            for i in range(num_samples)
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                results.append(result)
            progress.update(task, advance=1)

    if not results:
        console.print("[red]No samples generated successfully.[/red]")
        return

    # Save to parquet
    text_dir = output_dir / "text_only"
    text_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "sample_id": [r["sample_id"] for r in results],
            "persona": [r["persona"] for r in results],
            "tools_json": [r["tools_json"] for r in results],
            "conversation": [r["conversation"] for r in results],
            "num_turns": [r["num_turns"] for r in results],
            "num_tool_calls": [r["num_tool_calls"] for r in results],
        },
        schema=PARQUET_SCHEMA,
    )

    output_path = text_dir / "conversations.parquet"
    pq.write_table(table, output_path, compression="zstd")

    console.print(
        f"[green]Generated {len(results)}/{num_samples} samples -> {output_path}[/green]"
    )


def _synthesize_sample(
    sample: dict,
    voice: str,
) -> dict | None:
    """Synthesize audio for all user turns in a single sample."""
    from tts import synthesize_speech

    conversation = json.loads(sample["conversation"])
    user_audio: list[bytes] = []

    for turn in conversation:
        if turn["role"] == "user":
            text = turn.get("content", "")
            if not text.strip():
                user_audio.append(b"")
                continue
            try:
                wav_bytes = synthesize_speech(text, voice)
                user_audio.append(wav_bytes)
            except Exception as e:
                console.print(
                    f"[yellow]TTS failed for sample {sample['sample_id']}: {e}[/yellow]"
                )
                return None

    return {
        **sample,
        "voice": voice,
        "user_audio": user_audio,
    }


@app.command()
def generate(
    num_samples: int = typer.Option(100, help="Number of samples to generate"),
    batch_size: int = typer.Option(16, help="Concurrent LLM requests"),
    output_dir: Path = typer.Option(Path("output"), help="Output directory"),
) -> None:
    """Generate synthetic tool-calling conversations (text only)."""
    console.print(
        f"Generating {num_samples} samples with batch_size={batch_size}..."
    )
    asyncio.run(_generate_all(num_samples, batch_size, output_dir))


@app.command()
def generate_audio(
    input_dir: Path = typer.Option(
        Path("output/text_only"), help="Directory with text-only parquet files"
    ),
    output_dir: Path = typer.Option(
        Path("output/audio"), help="Output directory for audio parquet"
    ),
    batch_size: int = typer.Option(8, help="Concurrent TTS requests"),
) -> None:
    """Generate audio for user turns from existing text-only conversations."""
    from tts import pick_voice

    parquet_files = list(input_dir.glob("*.parquet"))
    # Exclude labels.parquet
    parquet_files = [f for f in parquet_files if f.name != "labels.parquet"]
    if not parquet_files:
        console.print(f"[red]No parquet files found in {input_dir}[/red]")
        raise typer.Exit(1)

    # Load samples
    samples: list[dict] = []
    for f in parquet_files:
        table = pq.read_table(f)
        for i in range(len(table)):
            samples.append({col: table[col][i].as_py() for col in table.column_names})

    if not samples:
        console.print("[red]No samples found.[/red]")
        raise typer.Exit(1)

    results: list[dict] = []

    # Assign a voice per sample
    sample_voices = [(s, pick_voice()) for s in samples]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesizing audio...", total=len(samples))

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(_synthesize_sample, sample, voice): sample
                for sample, voice in sample_voices
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                progress.update(task, advance=1)

    if not results:
        console.print("[red]No audio samples generated successfully.[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "sample_id": [r["sample_id"] for r in results],
            "persona": [r["persona"] for r in results],
            "tools_json": [r["tools_json"] for r in results],
            "conversation": [r["conversation"] for r in results],
            "num_turns": [r["num_turns"] for r in results],
            "num_tool_calls": [r["num_tool_calls"] for r in results],
            "voice": [r["voice"] for r in results],
            "user_audio": [r["user_audio"] for r in results],
        },
        schema=AUDIO_PARQUET_SCHEMA,
    )

    output_path = output_dir / "conversations.parquet"
    pq.write_table(table, output_path, compression="zstd")

    console.print(
        f"[green]Generated audio for {len(results)}/{len(samples)} samples -> {output_path}[/green]"
    )


@app.command()
def inspect(
    input_dir: Path = typer.Option(
        Path("output/text_only"), help="Directory with parquet files"
    ),
) -> None:
    """Launch the conversation inspector."""
    from inspect_data import run_inspector

    run_inspector(input_dir)


if __name__ == "__main__":
    app()
