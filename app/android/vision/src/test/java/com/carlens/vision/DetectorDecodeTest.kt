package com.carlens.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectorDecodeTest {

    private val контракт = DetectorContract(
        imgsz = 416,
        inputDtype = "float32",
        inputLayout = "NCHW",
        channelOrder = "RGB",
        inputRange = "unit",
        letterboxPadValue = 114,
        outputLayout = "CN",
        boxFormat = "xywh",
        coordUnits = "normalized",
        numClasses = 80,
        vehicleClasses = listOf(2, 5, 7),
        scoreLayout = "classes_only",
        maxDet = 100,
    )

    private val каналов = 84
    private val якорей = 6
    private val отображение = Letterbox.forFrame(UprightSize(416, 416), 416)

    /** Выход `[1, 84, N]`: канал `c` якоря `n` лежит в `c * N + n`. */
    private fun заполнитьCN(
        сырой: FloatArray,
        якорь: Int,
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        класс: Int,
        скор: Float,
    ) {
        сырой[0 * якорей + якорь] = cx
        сырой[1 * якорей + якорь] = cy
        сырой[2 * якорей + якорь] = w
        сырой[3 * якорей + якорь] = h
        сырой[(4 + класс) * якорей + якорь] = скор
    }

    private fun разобрать(
        сырой: FloatArray,
        contract: DetectorContract = контракт,
    ): List<Box> = DetectorDecode.boxes(
        сырой, contract, каналов, якорей, единицы = contract.imgsz.toFloat(), отображение = отображение,
    )

    @Test
    fun `машина выше порога попадает в ответ`() {
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, якорь = 0, cx = 0.5f, cy = 0.5f, w = 0.2f, h = 0.1f, класс = 2, скор = 0.9f)
        val рамки = разобрать(сырой)
        assertEquals(1, рамки.size)
        assertEquals(0.9f, рамки[0].conf, 1e-6f)
        assertEquals(416f * (0.5f - 0.1f), рамки[0].x, 1e-3f)
        assertEquals(416f * 0.2f, рамки[0].w, 1e-3f)
    }

    @Test
    fun `скор ниже порога отбрасывается`() {
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, 0, 0.5f, 0.5f, 0.2f, 0.1f, класс = 2, скор = 0.34f)
        assertTrue(разобрать(сырой).isEmpty())
    }

    @Test
    fun `не транспортные классы игнорируются даже с высоким скором`() {
        // Класс 0 — человек: уверенная рамка на нём не должна доехать до кропа.
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, 0, 0.5f, 0.5f, 0.2f, 0.1f, класс = 0, скор = 0.99f)
        assertTrue(разобрать(сырой).isEmpty())
    }

    @Test
    fun `все три класса транспорта проходят`() {
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, 0, 0.1f, 0.1f, 0.05f, 0.05f, класс = 2, скор = 0.9f)
        заполнитьCN(сырой, 1, 0.5f, 0.5f, 0.05f, 0.05f, класс = 5, скор = 0.8f)
        заполнитьCN(сырой, 2, 0.9f, 0.9f, 0.05f, 0.05f, класс = 7, скор = 0.7f)
        val рамки = разобрать(сырой)
        assertEquals(3, рамки.size)
        assertEquals(listOf(0.9f, 0.8f, 0.7f), рамки.map { it.conf })
    }

    @Test
    fun `берётся лучший класс якоря, а не первый по счёту`() {
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, 0, 0.5f, 0.5f, 0.2f, 0.1f, класс = 2, скор = 0.4f)
        сырой[(4 + 7) * якорей + 0] = 0.85f
        assertEquals(0.85f, разобрать(сырой)[0].conf, 1e-6f)
    }

    @Test
    fun `перекрывающиеся рамки схлопываются NMS`() {
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, 0, 0.5f, 0.5f, 0.2f, 0.2f, класс = 2, скор = 0.9f)
        заполнитьCN(сырой, 1, 0.51f, 0.51f, 0.2f, 0.2f, класс = 2, скор = 0.7f)
        val рамки = разобрать(сырой)
        assertEquals(1, рамки.size)
        assertEquals(0.9f, рамки[0].conf, 1e-6f)
    }

    @Test
    fun `NC даёт те же рамки, что CN`() {
        // Перепутанная раскладка выхода не падает, а выдаёт мусорные координаты: обе
        // ветки индексации обязаны читать один и тот же логический тензор одинаково.
        val cn = FloatArray(каналов * якорей)
        заполнитьCN(cn, 1, 0.4f, 0.6f, 0.1f, 0.2f, класс = 5, скор = 0.75f)
        val nc = FloatArray(каналов * якорей)
        for (якорь in 0 until якорей) {
            for (канал in 0 until каналов) nc[якорь * каналов + канал] = cn[канал * якорей + якорь]
        }
        val изCN = разобрать(cn)
        val изNC = разобрать(nc, контракт.copy(outputLayout = "NC"))
        assertEquals(изCN, изNC)
    }

    @Test
    fun `координаты в пикселях не масштабируются повторно`() {
        val контрактПиксели = контракт.copy(coordUnits = "pixels")
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, 0, 208f, 208f, 80f, 40f, класс = 2, скор = 0.9f)
        val рамки = DetectorDecode.boxes(
            сырой, контрактПиксели, каналов, якорей, единицы = 1f, отображение = отображение,
        )
        assertEquals(168f, рамки[0].x, 1e-3f)
        assertEquals(80f, рамки[0].w, 1e-3f)
    }

    @Test
    fun `maxDet режет ответ`() {
        val сырой = FloatArray(каналов * якорей)
        for (якорь in 0 until якорей) {
            заполнитьCN(сырой, якорь, 0.1f * (якорь + 1), 0.1f, 0.02f, 0.02f, класс = 2, скор = 0.5f + якорь / 100f)
        }
        val рамки = разобрать(сырой, контракт.copy(maxDet = 2))
        assertEquals(2, рамки.size)
    }

    @Test
    fun `пустой выход даёт пустой список, а не отказ по maxDet`() {
        assertTrue(разобрать(FloatArray(каналов * якорей)).isEmpty())
    }

    @Test
    fun `рамки разворачиваются из тензора в кадр через letterbox`() {
        // Кадр 832×416 вписан в тензор 416: масштаб 0.5, поля сверху и снизу по 104.
        val кадр = Letterbox.forFrame(UprightSize(832, 416), 416)
        val сырой = FloatArray(каналов * якорей)
        заполнитьCN(сырой, 0, 0.5f, 0.5f, 0.2f, 0.2f, класс = 2, скор = 0.9f)
        val рамки = DetectorDecode.boxes(сырой, контракт, каналов, якорей, 416f, кадр)
        val ожидаемыйX = (416f * 0.4f - кадр.padX) / кадр.scale
        val ожидаемыйY = (416f * 0.4f - кадр.padY) / кадр.scale
        assertEquals(ожидаемыйX, рамки[0].x, 1e-3f)
        assertEquals(ожидаемыйY, рамки[0].y, 1e-3f)
        assertEquals(416f * 0.2f / кадр.scale, рамки[0].w, 1e-3f)
    }
}
