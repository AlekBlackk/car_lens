from pipeline.harvest.admission import build_admit


def test_чужая_полка_не_допускается() -> None:
    """`Lada Niva Travel` — отдельный класс каталога, а не подполка `Lada Niva`."""
    допустить = build_admit(
        "lada__niva_2121",
        {"lada__niva_2121": ("niva",), "lada__niva_travel": ("niva travel",)},
        {"lada niva travel": frozenset({"lada__niva_travel"})},
    )

    assert допустить("Lada Niva Travel") is False


def test_своя_полка_допускается_по_игле() -> None:
    допустить = build_admit("bmw__3_series", {"bmw__3_series": ("e90",)}, {})

    assert допустить("BMW 318i (E90)") is True


def test_чужая_игла_не_допускается() -> None:
    допустить = build_admit(
        "geely__monjaro",
        {"geely__monjaro": ("monjaro",), "gaz__volga": ("volga",)},
        {},
    )

    assert допустить("Volga K50") is False


def test_имя_без_игл_не_допускается() -> None:
    """`unknown` бракуется так же жёстко, как `foreign`: молчащее имя не свидетельство."""
    допустить = build_admit("lada__2101", {"lada__2101": ("2101",)}, {})

    assert допустить("VAZ-21012") is False
