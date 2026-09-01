"""Реальные реализации протоколов: сеть, детекторы, энкодер.

Модуль намеренно тонкий и почти не покрыт тестами: всё, что можно проверить без весов
и без сети, вынесено в чистые функции (`decode_ssd_output`), а остальное — перевод
формата. Логика живёт в `run.py`, `redact.py` и `filters.py`, и она тестируется на
подставных объектах.

HTTP берётся из стандартной библиотеки: новый пакет в `pyproject.toml` ради трёх
запросов означал бы новую строку в пинах §13.4 и новую причину, по которой окружение
разъедется.
"""

from __future__ import annotations

import http.client
import urllib.error
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

from pipeline.crop import Detection
from pipeline.harvest.commons import DomainPacer, HttpResponse

USER_AGENT_TEMPLATE = "CarLens-harvest/0.1 (https://github.com/carlens; {contact})"
HTTP_TIMEOUT_SECONDS = 30


class UrllibTransport:
    """HTTP поверх стандартной библиотеки. Контакт в User-Agent обязателен (§3.2).

    Здесь же держится интервал между запросами — вторая половина того же §3.2. Место
    выбрано так потому, что это единственная точка, через которую запрос действительно
    уходит в сеть: поиск, скачивание и обход категорий все идут сюда, и вежливость
    перестаёт зависеть от того, вспомнил ли о ней очередной вызывающий. Пейсер приходит
    параметром, чтобы тест мог подставить свои часы и сон вместо настоящих.

    Логика интервала живёт в `DomainPacer` (`commons.py`), а не здесь: этот модуль
    намеренно тонкий и почти не покрыт тестами, а арифметику интервала проверять нужно.
    """

    def __init__(self, contact: str, pacer: DomainPacer | None = None) -> None:
        self._headers = {"User-Agent": USER_AGENT_TEMPLATE.format(contact=contact)}
        self._pacer = pacer if pacer is not None else DomainPacer()

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        адрес = f"{url}?{urlencode(dict(params))}" if params else url
        запрос = Request(адрес, headers=self._headers)
        self._pacer.wait(url)
        try:
            with urlopen(запрос, timeout=HTTP_TIMEOUT_SECONDS) as ответ:
                return HttpResponse(int(ответ.status), ответ.read())
        except (OSError, http.client.HTTPException, urllib.error.URLError, TimeoutError) as ошибка:
            # Код 429 и 503 приходят исключением: его разбирает `request_with_backoff`,
            # поэтому статус надо вернуть, а не потерять внутри трассировки.
            код = int(getattr(ошибка, "code", 0) or getattr(ошибка, "status", 0) or 0)
            return HttpResponse(код or 599, b"")
        finally:
            # Отметка и на провале: неудачный запрос хранилище всё равно обслужило, и
            # частить после ошибки — ровно то, чего просит не делать §3.2.
            self._pacer.mark(url)


def decode_ssd_output(raw: np.ndarray, frame_w: int, frame_h: int, conf: float) -> list[Detection]:
    """Разобрать выход SSD-детектора формы (1, 1, N, 7) в боксы исходного кадра."""
    боксы: list[Detection] = []
    for _, класс, уверенность, x0, y0, x1, y1 in raw.reshape(-1, 7):
        if float(уверенность) < conf:
            continue
        лx = max(0.0, float(x0) * frame_w)
        лy = max(0.0, float(y0) * frame_h)
        пx = min(float(frame_w), float(x1) * frame_w)
        пy = min(float(frame_h), float(y1) * frame_h)
        боксы.append(
            Detection(x=лx, y=лy, w=пx - лx, h=пy - лy, conf=float(уверенность), cls=int(класс))
        )
    return боксы


class OnnxBoxDetector:
    """Детектор номеров: ONNX через `cv2.dnn`, выход разбирается как SSD."""

    def __init__(self, model_path: Path, conf: float = 0.5, size: int = 300) -> None:
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._conf = conf
        self._size = size

    def detect(self, image: np.ndarray) -> list[Detection]:
        высота, ширина = image.shape[:2]
        blob = np.expand_dims(cv2.resize(image, (self._size, self._size)), axis=0).astype(
            np.float32
        )
        self._net.setInput(blob)
        return decode_ssd_output(self._net.forward(), ширина, высота, self._conf)


class YuNetFaceDetector:
    """Детектор лиц YuNet. Вход BGR — ровно то, в чём работает конвейер."""

    def __init__(self, model_path: Path, conf: float = 0.6) -> None:
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), conf)

    def detect(self, image: np.ndarray) -> list[Detection]:
        высота, ширина = image.shape[:2]
        self._detector.setInputSize((ширина, высота))
        _, лица = self._detector.detect(image)
        if лица is None:
            return []
        return [
            Detection(
                x=float(лицо[0]),
                y=float(лицо[1]),
                w=float(лицо[2]),
                h=float(лицо[3]),
                conf=float(лицо[14]),
                cls=0,
            )
            for лицо in лица
        ]


class ClipImageEncoder:
    """Энкодер изображений CLIP. Вход RGB, выход — нормированный вектор."""

    def __init__(self, model: Any, preprocess: Any) -> None:
        self._model = model
        self._preprocess = preprocess

    def encode_rgb(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        тензор = self._preprocess(Image.fromarray(image)).unsqueeze(0)
        with torch.no_grad():
            вектор = self._model.encode_image(тензор)
            вектор = вектор / вектор.norm(dim=-1, keepdim=True)
        итог: np.ndarray = вектор.squeeze(0).cpu().numpy().astype(np.float32)
        return итог


def build_text_matrix(names: Sequence[str], model: Any, tokenizer: Any) -> np.ndarray:
    """Матрица текстовых векторов по шаблону спайка: `a photo of a {name}, a car`."""
    import torch

    промпты = [f"a photo of a {имя}, a car" for имя in names]
    with torch.no_grad():
        векторы = model.encode_text(tokenizer(промпты))
        векторы = векторы / векторы.norm(dim=-1, keepdim=True)
    итог: np.ndarray = векторы.cpu().numpy().astype(np.float32)
    return итог
