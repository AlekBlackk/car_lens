package com.carlens.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GeometryTest {

    private val кадр = UprightSize(width = 1080, height = 1920)

    @Test
    fun `letterbox масштабирует по меньшей стороне и центрирует поля`() {
        val letterbox = Letterbox.forFrame(кадр, target = 416)
        assertEquals(416f / 1920f, letterbox.scale, 1e-6f)
        assertEquals(0f, letterbox.padY, 1e-4f)
        assertEquals((416f - 1080f * letterbox.scale) / 2f, letterbox.padX, 1e-4f)
    }

    @Test
    fun `прямое и обратное преобразование дают исходный бокс`() {
        val letterbox = Letterbox.forFrame(кадр, target = 416)
        val исходный = Box(x = 120f, y = 400f, w = 300f, h = 220f, conf = 0.8f)
        val туда = letterbox.uprightToTensor(исходный)
        val обратно = letterbox.tensorToUpright(туда)
        assertEquals(исходный.x, обратно.x, 1e-3f)
        assertEquals(исходный.y, обратно.y, 1e-3f)
        assertEquals(исходный.w, обратно.w, 1e-3f)
        assertEquals(исходный.h, обратно.h, 1e-3f)
    }

    @Test
    fun `пустой кадр и нулевая сторона тензора роняют с обоими числами`() {
        // Без этой проверки `scale` уходит в бесконечность или NaN, и наружу идут
        // правдоподобные рамки вместо отказа — глобальное ограничение вехи.
        val поКадру = assertThrows(IllegalArgumentException::class.java) {
            Letterbox.forFrame(UprightSize(width = 0, height = 1920), target = 416)
        }
        assertTrue(поКадру.message!!.contains("0"))
        assertTrue(поКадру.message!!.contains("1920"))
        assertThrows(IllegalArgumentException::class.java) {
            Letterbox.forFrame(кадр, target = 0)
        }
    }

    @Test
    fun `поворот на 90 и 270 меняет стороны местами`() {
        assertEquals(UprightSize(1920, 1080), Rotation.upright(1080, 1920, 90))
        assertEquals(UprightSize(1920, 1080), Rotation.upright(1080, 1920, 270))
        assertEquals(UprightSize(1080, 1920), Rotation.upright(1080, 1920, 0))
        assertEquals(UprightSize(1080, 1920), Rotation.upright(1080, 1920, 180))
    }

    @Test
    fun `точка выпрямленного кадра переводится в буфер сенсора на всех углах`() {
        for (угол in listOf(0, 90, 180, 270)) {
            val выпрямленный = Rotation.upright(1080, 1920, угол)
            val точка = Rotation.toSensor(10f, 20f, выпрямленный, угол)
            assertTrue("угол $угол: x вне буфера", точка[0] in 0f..1080f)
            assertTrue("угол $угол: y вне буфера", точка[1] in 0f..1920f)
        }
    }

    @Test
    fun `двойное выпрямление отличается от однократного`() {
        // Ловит ошибку «поворот применён и снаружи, и внутри»: она не падает, а даёт
        // правдоподобную рамку не на машине (§5.1 дизайна).
        val выпрямленный = Rotation.upright(1080, 1920, 90)
        val один = Rotation.toSensor(10f, 20f, выпрямленный, 90)
        val дважды = Rotation.toSensor(один[0], один[1], выпрямленный, 90)
        assertNotEquals(один[0], дважды[0], 1e-3f)
    }

    @Test
    fun `поворот не кратный 90 роняет`() {
        assertThrows(IllegalArgumentException::class.java) { Rotation.upright(1080, 1920, 45) }
    }

    @Test
    fun `паддинг 8 процентов расширяет бокс`() {
        val бокс = Box(x = 100f, y = 100f, w = 200f, h = 100f, conf = 0.9f)
        val расширенный = Boxes.pad(бокс, frameW = 1000, frameH = 1000, pad = 0.08f)
        assertEquals(100f - 200f * 0.08f, расширенный.x, 1e-3f)
        assertEquals(200f * 1.16f, расширенный.w, 1e-3f)
    }

    @Test
    fun `паддинг обрезается по краю кадра`() {
        val бокс = Box(x = 5f, y = 5f, w = 200f, h = 100f, conf = 0.9f)
        val расширенный = Boxes.pad(бокс, frameW = 300, frameH = 300, pad = 0.08f)
        assertEquals(0f, расширенный.x, 1e-6f)
        assertTrue(расширенный.x + расширенный.w <= 300f)
    }

    @Test
    fun `iou одинаковых боксов равен единице, непересекающихся нулю`() {
        val первый = Box(0f, 0f, 10f, 10f, 1f)
        assertEquals(1f, Boxes.iou(первый, первый), 1e-6f)
        assertEquals(0f, Boxes.iou(первый, Box(50f, 50f, 10f, 10f, 1f)), 1e-6f)
    }

    @Test
    fun `nms оставляет лучший из пересекающихся`() {
        val боксы = listOf(
            Box(0f, 0f, 100f, 100f, 0.9f),
            Box(5f, 5f, 100f, 100f, 0.7f),
            Box(500f, 500f, 50f, 50f, 0.6f),
        )
        val результат = Boxes.nms(боксы, iouThreshold = 0.5f, maxDet = 100)
        assertEquals(2, результат.size)
        assertEquals(0.9f, результат.first().conf, 1e-6f)
    }

    @Test
    fun `nms усекает по maxDet`() {
        val боксы = (0 until 10).map { Box(it * 200f, 0f, 50f, 50f, 0.5f + it / 100f) }
        assertEquals(3, Boxes.nms(боксы, iouThreshold = 0.5f, maxDet = 3).size)
    }

    @Test
    fun `nms на пустом входе даёт пустой выход`() {
        assertTrue(Boxes.nms(emptyList(), iouThreshold = 0.5f, maxDet = 10).isEmpty())
    }

    @Test
    fun `nms роняет бессмысленные maxDet и порог`() {
        val боксы = listOf(Box(0f, 0f, 10f, 10f, 0.9f))
        val поЧислу = assertThrows(IllegalArgumentException::class.java) {
            Boxes.nms(боксы, iouThreshold = 0.5f, maxDet = 0)
        }
        assertTrue(поЧислу.message!!.contains("0"))
        assertThrows(IllegalArgumentException::class.java) {
            Boxes.nms(боксы, iouThreshold = 1.5f, maxDet = 10)
        }
    }

    @Test
    fun `xywh в долях разворачивается в пиксели тензора`() {
        val бокс = Boxes.fromRaw(
            cx = 0.5f, cy = 0.25f, a = 0.1f, b = 0.2f,
            format = "xywh", unitsScale = 416f, conf = 0.8f,
        )
        assertEquals(416f * (0.5f - 0.05f), бокс.x, 1e-3f)
        assertEquals(416f * 0.1f, бокс.w, 1e-3f)
    }

    @Test
    fun `xyxy в пикселях разворачивается без масштабирования`() {
        val бокс = Boxes.fromRaw(
            cx = 10f, cy = 20f, a = 110f, b = 220f,
            format = "xyxy", unitsScale = 1f, conf = 0.8f,
        )
        assertEquals(10f, бокс.x, 1e-3f)
        assertEquals(100f, бокс.w, 1e-3f)
        assertEquals(200f, бокс.h, 1e-3f)
    }

    @Test
    fun `неизвестный формат бокса роняет с его названием`() {
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            Boxes.fromRaw(1f, 2f, 3f, 4f, format = "cxcywh", unitsScale = 1f, conf = 0.5f)
        }
        assertTrue(отказ.message!!.contains("cxcywh"))
    }
}
