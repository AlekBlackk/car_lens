import hashlib
from pathlib import Path

from spike.clip_zeroshot.fetch_models import sha256_of_file, write_weights_sha256
from spike.clip_zeroshot.manifest import load_models_manifest

ШАПКА = "name,repo_id,revision,url,license,checked,weights_sha256\n"
СТРОКА = (
    "tinyclip8m,wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,a2a8c6ea,"
    "https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M,mit,2026-08-26,\n"
)


def test_sha256_совпадает_с_hashlib(tmp_path: Path) -> None:
    файл = tmp_path / "weights.bin"
    файл.write_bytes(b"\x00\x01\x02" * 1000)
    assert sha256_of_file(файл) == hashlib.sha256(файл.read_bytes()).hexdigest()


def test_sha_записывается_не_ломая_остальные_столбцы(tmp_path: Path) -> None:
    путь = tmp_path / "models_manifest.csv"
    путь.write_text(ШАПКА + СТРОКА, encoding="utf-8")

    write_weights_sha256(путь, {"tinyclip8m": "a" * 64})

    пин = load_models_manifest(путь)[0]
    assert пин.weights_sha256 == "a" * 64
    assert пин.revision == "a2a8c6ea"
    assert пин.license == "mit"
