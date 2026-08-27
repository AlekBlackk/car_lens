"""Кодировщики четырёх кандидатов линейки за одним интерфейсом.

torch и библиотеки моделей импортируются внутри функций (Р-15): тесты семейства не
должны требовать ни группы `train`, ни сети.

Препроцессинг берётся тот, что отдаёт сама библиотека модели, а не наш собственный:
zero-shot чувствителен к нормализации и размеру, и подменять их своим кодом значило бы
мерить не модель, а нашу реализацию её препроцессинга.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from spike.clip_zeroshot.manifest import ModelPin


class EncoderError(ValueError):
    """Кодировщик не поднять."""


class Encoder(Protocol):
    """Что спайку нужно от модели: два энкодера, имя токенизатора и сам модуль.

    `model` объявлен в протоколе, а не достаётся `getattr`-ом по приватному имени:
    экспортёр фазы 2 оборачивает именно его, и обращение к нему — часть контракта,
    а не залезание внутрь.
    """

    tokenizer_name: str
    model: object

    def encode_text(self, промпты: list[str]) -> np.ndarray: ...

    def encode_images(self, пути: list[Path]) -> np.ndarray: ...


def family_of(пин: ModelPin) -> str:
    """Семейство по `repo_id`: от него зависит, чем грузить веса."""
    repo_id = пин.repo_id.lower()
    if "siglip" in repo_id:
        return "siglip"
    if "tinyclip" in repo_id or repo_id.startswith("laion/"):
        return "open_clip"
    raise EncoderError(f"неизвестное семейство для {repo_id!r}")


class _OpenClipEncoder:
    """TinyCLIP и OpenCLIP LAION — оба поднимаются через open_clip или transformers."""

    def __init__(self, пин: ModelPin, кэш: Path, device: str) -> None:
        import torch

        self._torch = torch
        self._device = device
        self._пин = пин

        # Проверяем, это TinyCLIP или обычный OpenCLIP
        это_tinyclip = "tinyclip" in пин.repo_id.lower()

        if это_tinyclip:
            # TinyCLIP загружается через transformers, но использует open_clip интерфейс
            from transformers import AutoModel, AutoProcessor

            self.model = (
                AutoModel.from_pretrained(пин.repo_id, revision=пин.revision, cache_dir=str(кэш))
                .to(device)
                .eval()
            )
            self._processor = AutoProcessor.from_pretrained(
                пин.repo_id, revision=пин.revision, cache_dir=str(кэш)
            )
            self.tokenizer_name = "transformers"
            self._используем_transformers = True
        else:
            # OpenCLIP LAION загружается через open_clip
            import open_clip

            модель, _, препроцесс = open_clip.create_model_and_transforms(
                f"hf-hub:{пин.repo_id}", cache_dir=str(кэш)
            )
            self.model = модель.to(device).eval()
            self._preprocess = препроцесс
            self._tokenize = open_clip.get_tokenizer(f"hf-hub:{пин.repo_id}")
            self.tokenizer_name = "open_clip"
            self._используем_transformers = False

    def encode_text(self, промпты: list[str]) -> np.ndarray:
        if self._используем_transformers:
            входы = self._processor(
                text=промпты, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self._device)
            with self._torch.no_grad():
                выход = self.model.get_text_features(**входы)
            вектора = выход.pooler_output
            return вектора.cpu().numpy().astype(np.float32)
        else:
            токены = self._tokenize(промпты).to(self._device)
            with self._torch.no_grad():
                вектора = self.model.encode_text(токены)
            return вектора.cpu().numpy().astype(np.float32)

    def encode_images(self, пути: list[Path]) -> np.ndarray:
        from PIL import Image

        куски: list[np.ndarray] = []
        for начало in range(0, len(пути), 32):
            пакет = пути[начало : начало + 32]

            if self._используем_transformers:
                изображения = [Image.open(п).convert("RGB") for п in пакет]
                входы = self._processor(images=изображения, return_tensors="pt").to(self._device)
                with self._torch.no_grad():
                    выход = self.model.get_image_features(**входы)
                вектора = выход.pooler_output
            else:
                тензоры = self._torch.stack(
                    [self._preprocess(Image.open(п).convert("RGB")) for п in пакет]
                ).to(self._device)
                with self._torch.no_grad():
                    вектора = self.model.encode_image(тензоры)

            куски.append(вектора.cpu().numpy().astype(np.float32))
        return np.concatenate(куски)


class _SiglipEncoder:
    """SigLIP2 поднимается `transformers`: своего пути в `open_clip` у него нет."""

    def __init__(self, пин: ModelPin, кэш: Path, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._device = device
        self.model = (
            AutoModel.from_pretrained(пин.repo_id, revision=пин.revision, cache_dir=str(кэш))
            .to(device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(
            пин.repo_id, revision=пин.revision, cache_dir=str(кэш)
        )
        self.tokenizer_name = "siglip"

    def encode_text(self, промпты: list[str]) -> np.ndarray:
        входы = self._processor(
            text=промпты, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self._device)
        with self._torch.no_grad():
            выход = self.model.get_text_features(**входы)
        вектора = выход.pooler_output
        return вектора.cpu().numpy().astype(np.float32)

    def encode_images(self, пути: list[Path]) -> np.ndarray:
        from PIL import Image

        куски: list[np.ndarray] = []
        for начало in range(0, len(пути), 32):
            пакет = [Image.open(п).convert("RGB") for п in пути[начало : начало + 32]]
            входы = self._processor(images=пакет, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                выход = self.model.get_image_features(**входы)
            вектора = выход.pooler_output
            куски.append(вектора.cpu().numpy().astype(np.float32))
        return np.concatenate(куски)


def load_encoder(пин: ModelPin, кэш: Path, device: str) -> Encoder:
    """Поднять кодировщик кандидата на приколоченной ревизии."""
    семейство = family_of(пин)
    if семейство == "open_clip":
        return _OpenClipEncoder(пин, кэш, device)
    return _SiglipEncoder(пин, кэш, device)
