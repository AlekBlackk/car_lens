from pathlib import Path

import pytest

from spike.clip_zeroshot.manifest import ManifestError, load_models_manifest

ШАПКА = "name,repo_id,revision,url,license,checked,weights_sha256\n"


def _записать(tmp_path: Path, тело: str) -> Path:
    путь = tmp_path / "models_manifest.csv"
    путь.write_text(ШАПКА + тело, encoding="utf-8")
    return путь


def test_полная_строка_читается(tmp_path: Path) -> None:
    путь = _записать(
        tmp_path,
        "tinyclip8m,wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,a2a8c6ea,"
        "https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,mit,2026-08-26,\n",
    )
    пины = load_models_manifest(путь)
    assert len(пины) == 1
    assert пины[0].repo_id == "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
    assert пины[0].revision == "a2a8c6ea"
    assert пины[0].weights_sha256 == ""


def test_пустая_ревизия_отвергается(tmp_path: Path) -> None:
    """Пин без ревизии не пин: карточка модели правится, и «проверено» ничего не значит."""
    путь = _записать(
        tmp_path,
        "tinyclip8m,wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,,"
        "https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,mit,2026-08-26,\n",
    )
    with pytest.raises(ManifestError, match="revision"):
        load_models_manifest(путь)


def test_пустая_лицензия_отвергается(tmp_path: Path) -> None:
    путь = _записать(
        tmp_path,
        "tinyclip8m,wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,a2a8c6ea,"
        "https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,,2026-08-26,\n",
    )
    with pytest.raises(ManifestError, match="license"):
        load_models_manifest(путь)


def test_повтор_имени_отвергается(tmp_path: Path) -> None:
    строка = (
        "tinyclip8m,wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,a2a8c6ea,"
        "https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,mit,2026-08-26,\n"
    )
    путь = _записать(tmp_path, строка + строка)
    with pytest.raises(ManifestError, match="повтор"):
        load_models_manifest(путь)


def test_реальный_манифест_проекта_полон() -> None:
    """Четыре кандидата линейки §4 спеки, у каждого приколочена ревизия."""
    from spike.clip_zeroshot.manifest import DEFAULT_MODELS_MANIFEST

    пины = load_models_manifest(DEFAULT_MODELS_MANIFEST)
    assert len(пины) == 4
    assert {п.name for п in пины} == {"tinyclip8m", "tinyclip39m", "openclip_b32", "siglip2_b16"}
    for пин in пины:
        assert len(пин.revision) == 40, f"{пин.name}: ревизия должна быть полным SHA"
        assert пин.license in {"mit", "apache-2.0"}
