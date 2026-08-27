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


class _TransformersEncoder:
    """Приватный базовый класс для моделей, загружаемых через transformers.

    Содержит общую логику для TinyCLIP и SigLIP2: оба используют
    AutoModel/AutoProcessor и get_*_features().pooler_output.
    """

    def __init__(self, пин: ModelPin, кэш: Path, device: str, имя_токенизатора: str) -> None:
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
        self.tokenizer_name = имя_токенизатора

    def encode_text(self, промпты: list[str]) -> np.ndarray:
        # У TinyCLIP токенизатор не задаёт свой `model_max_length` (в конфиге стоит
        # заглушка-«бесконечность»), и `padding="max_length"` без явной длины валится
        # с рваным батчем. Берём длину из конфига самой модели — она уже соответствует
        # размеру её позиционных эмбеддингов (77 у CLIP-семейства, 64 у SigLIP2).
        максимальная_длина = self.model.config.text_config.max_position_embeddings
        входы = self._processor(
            text=промпты,
            padding="max_length",
            truncation=True,
            max_length=максимальная_длина,
            return_tensors="pt",
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


class _TinyClipEncoder(_TransformersEncoder):
    """TinyCLIP — CLIP-модель через transformers."""

    def __init__(self, пин: ModelPin, кэш: Path, device: str) -> None:
        super().__init__(пин, кэш, device, "transformers")


class _OpenClipEncoder:
    """OpenCLIP LAION загружается через open_clip.

    Внимание: семейство open_clip используется для TinyCLIP и OpenCLIP LAION,
    но `tokenizer_name` у них РАЗНЫЙ (see load_encoder). TinyCLIP на самом деле
    это transformers CLIP-модель, а не настоящий OpenCLIP.
    """

    def __init__(self, пин: ModelPin, кэш: Path, device: str) -> None:
        import open_clip
        import torch

        self._torch = torch
        self._device = device
        модель, _, препроцесс = open_clip.create_model_and_transforms(
            f"hf-hub:{пин.repo_id}", cache_dir=str(кэш)
        )
        self.model = модель.to(device).eval()
        self._preprocess = препроцесс
        self._tokenize = open_clip.get_tokenizer(f"hf-hub:{пин.repo_id}")
        self.tokenizer_name = "open_clip"

    def encode_text(self, промпты: list[str]) -> np.ndarray:
        токены = self._tokenize(промпты).to(self._device)
        with self._torch.no_grad():
            вектора = self.model.encode_text(токены)
        return вектора.cpu().numpy().astype(np.float32)

    def encode_images(self, пути: list[Path]) -> np.ndarray:
        from PIL import Image

        куски: list[np.ndarray] = []
        for начало in range(0, len(пути), 32):
            пакет = пути[начало : начало + 32]
            тензоры = self._torch.stack(
                [self._preprocess(Image.open(п).convert("RGB")) for п in пакет]
            ).to(self._device)
            with self._torch.no_grad():
                вектора = self.model.encode_image(тензоры)
            куски.append(вектора.cpu().numpy().astype(np.float32))
        return np.concatenate(куски)


class _SiglipEncoder(_TransformersEncoder):
    """SigLIP2 через transformers."""

    def __init__(self, пин: ModelPin, кэш: Path, device: str) -> None:
        super().__init__(пин, кэш, device, "siglip")


def load_encoder(пин: ModelPin, кэш: Path, device: str) -> Encoder:
    """Поднять кодировщик кандидата на приколоченной ревизии.

    Семейство определяется family_of(), но внутри мы дополнительно проверяем,
    это TinyCLIP или обычный OpenCLIP LAION, несмотря на то что оба в семействе
    "open_clip". tokenizer_name результирующего кодировщика разный:
    - TinyCLIP: "transformers" (это transformers CLIP, не настоящий open_clip)
    - OpenCLIP LAION: "open_clip" (настоящий open_clip через hf-hub)
    """
    семейство = family_of(пин)
    if семейство == "open_clip":
        # Проверяем, это TinyCLIP или OpenCLIP LAION
        если_tinyclip = "tinyclip" in пин.repo_id.lower()
        if если_tinyclip:
            return _TinyClipEncoder(пин, кэш, device)
        return _OpenClipEncoder(пин, кэш, device)
    return _SiglipEncoder(пин, кэш, device)
