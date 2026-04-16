"""Human-in-the-loop inspection of generated conversations."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer()
console = Console()

# Audio players to try, in order of preference
_AUDIO_PLAYERS = ["mpv", "ffplay", "paplay", "aplay", "play"]


def _find_audio_player() -> str | None:
    """Find an available audio player on the system."""
    for player in _AUDIO_PLAYERS:
        if shutil.which(player):
            return player
    return None


def _play_audio(wav_bytes: bytes, player: str) -> None:
    """Play WAV audio bytes using a system audio player."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        f.write(wav_bytes)
        f.flush()
        try:
            args = [player, f.name]
            if player == "ffplay":
                args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", f.name]
            elif player == "mpv":
                args = [player, "--no-video", "--really-quiet", f.name]
            subprocess.run(args, check=True, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass
        except subprocess.CalledProcessError:
            console.print("[yellow]Audio playback failed[/yellow]")


def _display_conversation(
    sample: dict,
    index: int,
    total: int,
    audio_player: str | None,
) -> None:
    """Display a single conversation with rich formatting."""
    console.print(f"\n[bold]Sample {index + 1}/{total}[/bold] (id: {sample['sample_id']})")

    has_audio = "user_audio" in sample and sample["user_audio"]
    if sample.get("voice"):
        console.print(f"[dim]Voice: {sample['voice']}[/dim]")
    console.print()

    # Persona
    console.print(Panel(sample["persona"], title="Persona", border_style="blue"))

    # Tools
    tools = json.loads(sample["tools_json"])
    tool_table = Table(title="Available Tools", show_lines=True)
    tool_table.add_column("Name", style="cyan")
    tool_table.add_column("Description", style="white")
    tool_table.add_column("Parameters", style="dim")
    for tool in tools:
        fn = tool["function"]
        params = list(fn.get("parameters", {}).get("properties", {}).keys())
        tool_table.add_row(fn["name"], fn["description"], ", ".join(params))
    console.print(tool_table)
    console.print()

    # Conversation turns
    conversation = json.loads(sample["conversation"])
    user_audio_list = sample.get("user_audio") or []
    user_turn_idx = 0

    for turn in conversation:
        role = turn["role"]
        content = turn.get("content", "")

        if role == "user":
            style = "green"
            label = "User"
        elif role == "tool":
            style = "magenta"
            label = "Tool"
        else:
            style = "yellow"
            label = "Assistant"

        text = Text()
        text.append(f"[{label}] ", style=f"bold {style}")
        text.append(content)
        console.print(text)

        # Play audio for user turns if available
        if role == "user" and user_turn_idx < len(user_audio_list):
            audio_bytes = user_audio_list[user_turn_idx]
            if audio_bytes and audio_player:
                console.print("  [dim]Playing audio...[/dim]")
                _play_audio(audio_bytes, audio_player)
            elif audio_bytes:
                console.print("  [dim](audio available)[/dim]")
            user_turn_idx += 1

        # Show tool calls if present
        tool_calls = turn.get("tool_calls", [])
        for tc in tool_calls:
            fn_name = tc.get("function_name", tc.get("name", "unknown"))
            args = tc.get("arguments", {})
            console.print(
                f"  [dim]-> {fn_name}({json.dumps(args, indent=2)})[/dim]"
            )
        console.print()

    console.print(
        f"[dim]Turns: {sample['num_turns']} | Tool calls: {sample['num_tool_calls']}[/dim]"
    )


def run_inspector(input_dir: Path) -> None:
    """Run the interactive conversation inspector."""
    parquet_files = list(input_dir.glob("*.parquet"))
    # Exclude labels file
    parquet_files = [f for f in parquet_files if f.name != "labels.parquet"]
    if not parquet_files:
        console.print(f"[red]No parquet files found in {input_dir}[/red]")
        return

    # Load all samples
    samples: list[dict] = []
    for f in parquet_files:
        table = pq.read_table(f)
        for i in range(len(table)):
            samples.append({col: table[col][i].as_py() for col in table.column_names})

    if not samples:
        console.print("[red]No samples found.[/red]")
        return

    # Check for audio player
    audio_player = _find_audio_player()
    has_audio = "user_audio" in samples[0] and samples[0]["user_audio"]
    if has_audio and audio_player:
        console.print(f"[dim]Audio player: {audio_player}[/dim]")
    elif has_audio:
        console.print("[yellow]No audio player found. Install mpv, ffplay, or aplay for playback.[/yellow]")

    # Track labels
    labels: dict[str, str] = {}  # sample_id -> "accepted" | "rejected"
    index = 0

    commands = "[n]ext [p]rev [a]ccept [r]eject"
    if has_audio:
        commands += " [l]isten"
    commands += " [q]uit"
    console.print(f"[bold]Loaded {len(samples)} samples. Commands: {commands}[/bold]")

    while True:
        _display_conversation(samples[index], index, len(samples), audio_player=None)

        label = labels.get(samples[index]["sample_id"], "unlabeled")
        console.print(f"[bold]Status: {label}[/bold]")

        prompt = "[n/p/a/r"
        if has_audio:
            prompt += "/l"
        prompt += "/q] > "
        action = console.input(prompt).strip().lower()

        if action in ("n", ""):
            index = min(index + 1, len(samples) - 1)
        elif action == "p":
            index = max(index - 1, 0)
        elif action == "a":
            labels[samples[index]["sample_id"]] = "accepted"
            console.print("[green]Accepted[/green]")
            index = min(index + 1, len(samples) - 1)
        elif action == "r":
            labels[samples[index]["sample_id"]] = "rejected"
            console.print("[red]Rejected[/red]")
            index = min(index + 1, len(samples) - 1)
        elif action == "l" and has_audio:
            if audio_player:
                # Replay conversation with audio playback
                _display_conversation(samples[index], index, len(samples), audio_player=audio_player)
            else:
                console.print("[yellow]No audio player found. Install mpv, ffplay, or paplay.[/yellow]")
        elif action == "q":
            break

    # Save labels
    if labels:
        label_table = pa.table(
            {
                "sample_id": list(labels.keys()),
                "label": list(labels.values()),
            }
        )
        label_path = input_dir / "labels.parquet"
        pq.write_table(label_table, label_path)
        console.print(f"[green]Saved {len(labels)} labels to {label_path}[/green]")


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("output/text_only"), help="Directory with parquet files"
    ),
) -> None:
    """Inspect generated conversations with accept/reject labeling."""
    run_inspector(input_dir)


if __name__ == "__main__":
    app()
