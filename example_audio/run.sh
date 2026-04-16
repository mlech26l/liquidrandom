  # 1. Generate text conversations
  python generate.py generate --num-samples 256 --batch-size 32

  # 2. Synthesize audio for user turns
  python generate.py generate-audio --batch-size 8
