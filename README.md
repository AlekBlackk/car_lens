# CarLens

Распознавание марки и модели автомобиля с камеры телефона, полностью on-device.

Спецификация: [docs/SPEC.md](docs/SPEC.md). Решения на старте:
[docs/superpowers/specs/2026-08-15-carlens-decisions.md](docs/superpowers/specs/2026-08-15-carlens-decisions.md).

## Окружение

Питоновский конвейер работает в WSL2 Ubuntu 24.04, приложение собирается
на Windows (§16.1 спецификации). Драйвер NVIDIA ставится **только на
Windows** — установка драйвера внутри WSL ломает проброс GPU.

```bash
uv sync --group dev --group train
```

Каталог данных задаётся переменной `CARLENS_DATA_ROOT`, по умолчанию
`~/carlens-data`. Держать его в файловой системе WSL, не в `/mnt/c`.

## Проверки

```bash
uv run ruff check . && uv run mypy && uv run pytest
```

## Зафиксированные версии

Точные версии живут в `uv.lock`. Ключевые пины и причины — §13.4
спецификации.

| Пакет | Версия | Почему зафиксирован |
|---|---|---|
| Python | 3.11 | Нижняя граница `litert-torch` — 3.10; 3.11 даёт запас и стабильные колёса |
| torch | 2.12.0 | `litert-torch` 0.9.3 требует `>=2.4.0,<2.13.0` |
| torchvision | 0.27.0 | Парная к torch 2.12 |
| timm | 1.0.28 | Источник весов MobileNetV4 |
| litert-torch | 0.9.3 | Конвертация в TFLite; преемник `ai-edge-torch` |
| Node | 20.18.0 | Требование React Native 0.76 |
