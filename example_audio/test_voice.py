from pathlib import Path

from tts import VOICES, synthesize_speech

output_dir = Path(__file__).parent / "debug_output"
output_dir.mkdir(exist_ok=True)

text = "The quick brown fox jumped over the lazy dog."

for voice in VOICES:
    output_path = output_dir / f"{voice}.wav"
    print(f"Generating {voice}...")
    try:
        wav_bytes = synthesize_speech(text, voice)
        output_path.write_bytes(wav_bytes)
        print(f"  -> {output_path}")
    except Exception as e:
        print(f"  FAILED: {e}")
