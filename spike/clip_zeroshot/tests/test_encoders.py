from pathlib import Path

import pytest

from spike.clip_zeroshot.encoders import EncoderError, family_of
from spike.clip_zeroshot.manifest import ModelPin


def _пин(name: str, repo_id: str) -> ModelPin:
    return ModelPin(
        name=name,
        repo_id=repo_id,
        revision="a" * 40,
        url=f"https://huggingface.co/{repo_id}",
        license="mit",
        checked="2026-08-26",
        weights_sha256="",
    )


def test_семейство_по_repo_id() -> None:
    assert family_of(_пин("tinyclip8m", "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")) == "open_clip"
    assert family_of(_пин("openclip_b32", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K")) == "open_clip"
    assert family_of(_пин("siglip2_b16", "google/siglip2-base-patch16-256")) == "siglip"


def test_неизвестное_семейство_падает_внятно() -> None:
    with pytest.raises(EncoderError, match="неизвестное семейство"):
        family_of(_пин("что_то", "example/unknown-model"))


def test_загрузка_неизвестного_семейства_не_доходит_до_сети(tmp_path: Path) -> None:
    from spike.clip_zeroshot.encoders import load_encoder

    with pytest.raises(EncoderError, match="неизвестное семейство"):
        load_encoder(_пин("что_то", "example/unknown-model"), tmp_path, "cpu")
