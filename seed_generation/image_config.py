"""Configuration constants for image seed data generation."""

# Nano Banana 2 Lite: text-to-image + editing, 1K output only, 14 aspect ratios.
IMAGE_MODEL_NAME = "gemini-3.1-flash-lite-image"
# Cheap Gemini VLM used to validate generated images against their prompt/tags.
VALIDATION_MODEL_NAME = "gemini-flash-lite-latest"

DEFAULT_IMAGE_N = 20000
DEFAULT_IMAGE_K = 5  # chain plans per LLM call
DEFAULT_IMAGE_BATCH_SIZE = 16  # concurrent LLM (chain-plan) calls
DEFAULT_IMAGE_CONCURRENCY = 30  # concurrent Gemini image calls (global)
DEFAULT_IMAGE_TAXONOMY_DEPTH = 3
DEFAULT_IMAGE_SAMPLES_PER_LEAF = 80
DEFAULT_IMAGE_DEDUP_THRESHOLD = 0.7

# Chain shape: 1 base + EDITS_PER_BASE edits, edit types sampled without
# replacement per chain. Each edit after the first targets the previous
# image with probability EDIT_SEQUENTIAL_P, else the base image.
DEFAULT_EDITS_PER_BASE = 3
EDIT_SEQUENTIAL_P = 0.5
# Fraction of edited images checked by the VLM (bases are always checked).
EDIT_SPOT_CHECK_RATE = 0.1

MAX_IMAGE_RETRIES = 3
WEBP_QUALITY = 87
# Small row groups let the package read single random samples without
# materializing multi-GB files.
PARQUET_ROW_GROUP_SIZE = 64

# Preview mode: a handful of chains per category for human review
# (via `generate.py review-images`) before committing to the full run.
PREVIEW_N = 20
PREVIEW_SAMPLES_PER_LEAF = 4
PREVIEW_TAXONOMY_DEPTH = 2
