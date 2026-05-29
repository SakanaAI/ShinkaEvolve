from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import List, TypeAlias

from pydantic import BaseModel


SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}
SUPPORTED_IMAGE_RESOLUTIONS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
    "MEDIA_RESOLUTION_LOW": "MEDIA_RESOLUTION_LOW",
    "MEDIA_RESOLUTION_MEDIUM": "MEDIA_RESOLUTION_MEDIUM",
    "MEDIA_RESOLUTION_HIGH": "MEDIA_RESOLUTION_HIGH",
    "MEDIA_RESOLUTION_ULTRA_HIGH": "MEDIA_RESOLUTION_ULTRA_HIGH",
}

# Gemini supported image types: https://ai.google.dev/gemini-api/docs/image-understanding
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def _infer_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or _MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
        raise ValueError(
            f"Unsupported image file type for {path}. Supported MIME types: {supported}."
        )
    return mime_type


def _normalize_resolution(resolution: str | None) -> str | None:
    if resolution is None:
        return None
    normalized = SUPPORTED_IMAGE_RESOLUTIONS.get(resolution)
    if normalized is None:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_RESOLUTIONS))
        raise ValueError(
            f"Unsupported image resolution: {resolution!r}. Supported values: {supported}."
        )
    return normalized


class ImageInput(BaseModel):
    """Image input for an LLM call.

    ``mime_type`` is inferred from ``path``. ``resolution`` is optional and maps
    to Gemini's ``types.MediaResolution`` values when a Gemini request is built.
    """

    path: Path
    mime_type: str | None = None
    resolution: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "mime_type", _infer_mime_type(self.path))
        object.__setattr__(self, "resolution", _normalize_resolution(self.resolution))


ImageInputLike: TypeAlias = ImageInput | str | os.PathLike[str]
ImageInputs: TypeAlias = List[ImageInputLike] | None


def _coerce_image_input(image: ImageInputLike) -> ImageInput:
    if isinstance(image, ImageInput):
        return image
    return ImageInput(path=Path(image))


def load_image_inputs(images: ImageInputs) -> List[dict]:
    """Read image paths into provider-neutral payload dictionaries."""
    if images is None:
        return []

    image_payloads = []
    for image in images:
        image_input = _coerce_image_input(image)
        if not image_input.path.exists():
            raise FileNotFoundError(f"Image not found: {image_input.path}")
        image_payloads.append(
            {
                "data": image_input.path.read_bytes(),
                "mime_type": image_input.mime_type,
                "resolution": image_input.resolution,
            }
        )
    return image_payloads
