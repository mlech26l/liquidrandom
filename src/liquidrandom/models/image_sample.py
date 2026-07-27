from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class ImageSample:
    """A single generated image with its tags and edit-chain metadata.

    Shared by all image categories; ``category`` is injected by the loader
    (it is not a Parquet column). Images belonging to the same ``chain_id``
    form a base image (turn_index 0) plus a sequence of edited variants.

    Within a chain, ``turn_index`` runs 0..chain_length-1 and is the sample's
    position in the list returned by :func:`liquidrandom.image_chain`, so
    ``chain[sample.parent_turn]`` is the image this one was edited from
    (``parent_turn`` is -1 for the base). ``caption`` and ``prompt`` describe
    the base image and are repeated across the chain; ``edit_instruction``
    is what distinguishes an edited variant.
    """

    image: bytes = field(repr=False)
    image_format: str
    width: int
    height: int
    aspect_ratio: str
    category: str
    taxonomy_path: str
    caption: str
    prompt: str
    edit_instruction: str
    tags: list[str]
    chain_id: str
    turn_index: int
    parent_turn: int
    chain_length: int

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("category", "caption", "tags"),
        "detailed": (
            "taxonomy_path",
            "width",
            "height",
            "aspect_ratio",
            "chain_id",
            "turn_index",
            "edit_instruction",
        ),
    }

    @property
    def is_base(self) -> bool:
        return self.turn_index == 0

    @property
    def mime_type(self) -> str:
        return f"image/{self.image_format}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageSample:
        return cls(
            image=bytes(data["image"] or b""),
            image_format=data["image_format"],
            width=int(data["width"]),
            height=int(data["height"]),
            aspect_ratio=data["aspect_ratio"],
            category=data["category"],
            taxonomy_path=data["taxonomy_path"],
            caption=data["caption"],
            prompt=data["prompt"],
            edit_instruction=data["edit_instruction"] or "",
            tags=list(data["tags"] or []),
            chain_id=data["chain_id"],
            turn_index=int(data["turn_index"]),
            parent_turn=int(data["parent_turn"]),
            chain_length=int(data["chain_length"]),
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        tags = ", ".join(self.tags)
        base = f"{self.category} image: {self.caption} Tags: {tags}."
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        role = (
            "base image"
            if self.is_base
            else f"edit {self.turn_index} of {self.chain_length - 1} "
            f"({self.edit_instruction})"
        )
        return (
            f"{base} Topic: {self.taxonomy_path}. "
            f"{self.width}x{self.height} ({self.aspect_ratio}) {self.image_format}. "
            f"Chain {self.chain_id}: {role}."
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()

    def save(self, path: str | Path) -> Path:
        """Write the raw image bytes to ``path`` and return it."""
        out = Path(path)
        out.write_bytes(self.image)
        return out

    def to_pil(self) -> Any:
        """Decode the image with pillow (requires the ``image`` extra)."""
        try:
            from io import BytesIO

            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "pillow is required for to_pil(); "
                "install with: pip install liquidrandom[image]"
            ) from e
        return Image.open(BytesIO(self.image))
