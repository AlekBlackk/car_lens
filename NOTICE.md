# Сторонние компоненты

Проект личный и учебный, сборки не распространяются. Полный разбор
ограничений и план замены при смене режима — §15 спецификации.

| Компонент | Лицензия | Примечание |
|---|---|---|
| ultralytics / yolo11n | AGPL-3.0 | Заменяется на YOLOX-Nano или NanoDet-Plus при любом распространении |
| Stanford Cars | Research-only | Только этап M-1, веса расходные |
| CompCars | Research-only | Только проверка архитектуры, в продакшн-веса не попадает |
| timm `mobilenetv4_conv_medium.e500_r256_in1k` | Apache-2.0 | Ограничений нет |
| Jordo23/vehicle-classifier `class_mapping.csv` | MIT | Требуется указание авторства |
| PyTorch, LiteRT, OpenCV, albumentations | BSD / Apache-2.0 | Ограничений нет |
