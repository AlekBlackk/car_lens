package com.carlens.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer

class FrameConvertTest {

    /** Ровный цвет по всему кадру, шаги как у реального устройства: uv чередуются. */
    private fun ровныйКадр(
        width: Int, height: Int, r: Int, g: Int, b: Int, rotation: Int = 0,
    ): YuvFrame {
        val яркость = (0.257 * r + 0.504 * g + 0.098 * b + 16).toInt()
        val синева = (-0.148 * r - 0.291 * g + 0.439 * b + 128).toInt()
        val краснота = (0.439 * r - 0.368 * g - 0.071 * b + 128).toInt()
        val yРяд = width + 16                       // rowStride больше ширины: выравнивание
        val uvРяд = width + 16
        val yБуфер = ByteBuffer.allocate(yРяд * height)
        val uБуфер = ByteBuffer.allocate(uvРяд * (height / 2))
        val vБуфер = ByteBuffer.allocate(uvРяд * (height / 2))
        for (строка in 0 until height) {
            for (столбец in 0 until width) yБуфер.put(строка * yРяд + столбец, яркость.toByte())
        }
        for (строка in 0 until height / 2) {
            for (столбец in 0 until width / 2) {
                uБуфер.put(строка * uvРяд + столбец * 2, синева.toByte())
                vБуфер.put(строка * uvРяд + столбец * 2, краснота.toByte())
            }
        }
        return YuvFrame(
            width = width, height = height, rotationDegrees = rotation,
            y = yБуфер, yRowStride = yРяд, yPixelStride = 1,
            u = uБуфер, v = vБуфер, uvRowStride = uvРяд, uvPixelStride = 2,
        )
    }

    private val спекДетектора = TensorSpec(
        size = 8, layout = TensorLayout.NHWC, channels = ChannelOrder.RGB, scaling = Scaling.Unit01,
    )

    @Test
    fun `ровный красный кадр даёт красный тензор`() {
        val кадр = ровныйКадр(16, 16, r = 220, g = 30, b = 30)
        val тензор = FloatArray(спекДетектора.floatCount)
        FrameConvert.letterbox(кадр, спекДетектора, padValue = 114, dst = тензор)
        assertEquals(220f / 255f, тензор[0], 0.03f)
        assertEquals(30f / 255f, тензор[1], 0.03f)
        assertEquals(30f / 255f, тензор[2], 0.03f)
    }

    @Test
    fun `порядок каналов BGR переставляет крайние`() {
        val кадр = ровныйКадр(16, 16, r = 220, g = 30, b = 30)
        val спек = спекДетектора.copy(channels = ChannelOrder.BGR)
        val тензор = FloatArray(спек.floatCount)
        FrameConvert.letterbox(кадр, спек, padValue = 114, dst = тензор)
        assertEquals(30f / 255f, тензор[0], 0.03f)
        assertEquals(220f / 255f, тензор[2], 0.03f)
    }

    @Test
    fun `поля letterbox заполнены серым 114`() {
        val кадр = ровныйКадр(16, 8, r = 200, g = 200, b = 200)   // широкий кадр: поля сверху и снизу
        val тензор = FloatArray(спекДетектора.floatCount)
        FrameConvert.letterbox(кадр, спекДетектора, padValue = 114, dst = тензор)
        assertEquals(114f / 255f, тензор[0], 1e-3f)               // верхний левый пиксель — поле
    }

    @Test
    fun `letterbox возвращает то же отображение, что Letterbox_forFrame`() {
        val кадр = ровныйКадр(16, 8, r = 10, g = 10, b = 10, rotation = 90)
        val тензор = FloatArray(спекДетектора.floatCount)
        val отображение = FrameConvert.letterbox(кадр, спекДетектора, padValue = 114, dst = тензор)
        assertEquals(Letterbox.forFrame(кадр.upright, спекДетектора.size), отображение)
    }

    @Test
    fun `NCHW раскладывает канал целиком перед следующим`() {
        val кадр = ровныйКадр(16, 16, r = 255, g = 0, b = 0)
        val спек = спекДетектора.copy(layout = TensorLayout.NCHW)
        val тензор = FloatArray(спек.floatCount)
        FrameConvert.letterbox(кадр, спек, padValue = 114, dst = тензор)
        val пикселей = спек.size * спек.size
        assertTrue(тензор[0] > 0.9f)                     // красный канал
        assertTrue(тензор[пикселей] < 0.1f)              // зелёный
    }

    @Test
    fun `нормализация mean-std применяется поканально`() {
        val кадр = ровныйКадр(16, 16, r = 128, g = 128, b = 128)
        val спек = TensorSpec(
            size = 8, layout = TensorLayout.NHWC, channels = ChannelOrder.RGB,
            scaling = Scaling.MeanStd(
                mean = floatArrayOf(0.5f, 0.5f, 0.5f), std = floatArrayOf(0.25f, 0.25f, 0.25f),
            ),
        )
        val тензор = FloatArray(спек.floatCount)
        FrameConvert.letterbox(кадр, спек, padValue = 114, dst = тензор)
        assertEquals((128f / 255f - 0.5f) / 0.25f, тензор[0], 0.05f)
    }

    @Test
    fun `билинейная выборка даёт промежуточное значение на градиенте`() {
        // Градиент по x: столбец k имеет яркость 16 + k*8. Целевой пиксель между
        // столбцами обязан дать значение между ними, а не одно из них.
        val width = 16
        val height = 16
        val yРяд = width + 8
        val yБуфер = ByteBuffer.allocate(yРяд * height)
        for (строка in 0 until height) {
            for (столбец in 0 until width) {
                yБуфер.put(строка * yРяд + столбец, (16 + столбец * 8).toByte())
            }
        }
        val uvРяд = width
        val серый = ByteBuffer.allocate(uvРяд * (height / 2))
        for (индекс in 0 until серый.capacity()) серый.put(индекс, 128.toByte())
        val кадр = YuvFrame(
            width = width, height = height, rotationDegrees = 0,
            y = yБуфер, yRowStride = yРяд, yPixelStride = 1,
            u = серый, v = серый, uvRowStride = uvРяд, uvPixelStride = 1,
        )
        val спек = TensorSpec(
            size = 5, layout = TensorLayout.NHWC, channels = ChannelOrder.RGB,
            scaling = Scaling.Unit01,
        )
        val тензор = FloatArray(спек.floatCount)
        FrameConvert.letterbox(кадр, спек, padValue = 114, dst = тензор)
        val соседи = (0 until 5).map { тензор[it * 3] }
        assertTrue(
            "значения обязаны расти по x: $соседи",
            соседи.zipWithNext().all { it.first < it.second },
        )
    }

    @Test
    fun `выборка не сваливается в ближайшего соседа`() {
        // Ловит подмену билинейной выборки на ближайшего соседа: точка ровно между
        // двумя столбцами градиента обязана дать их полусумму, а не одно из значений.
        val width = 16
        val height = 16
        val yБуфер = ByteBuffer.allocate(width * height)
        for (строка in 0 until height) {
            for (столбец in 0 until width) {
                yБуфер.put(строка * width + столбец, (16 + столбец * 8).toByte())
            }
        }
        val серый = ByteBuffer.allocate(width * (height / 2))
        for (индекс in 0 until серый.capacity()) серый.put(индекс, 128.toByte())
        val кадр = YuvFrame(
            width = width, height = height, rotationDegrees = 0,
            y = yБуфер, yRowStride = width, yPixelStride = 1,
            u = серый, v = серый, uvRowStride = width, uvPixelStride = 1,
        )
        val спек = TensorSpec(
            size = 4, layout = TensorLayout.NHWC, channels = ChannelOrder.RGB,
            scaling = Scaling.Unit01,
        )
        val тензор = FloatArray(спек.floatCount)
        // Бокс шириной 4 на четыре столбца со сдвигом в полпикселя: центры выборки
        // попадают ровно между соседними столбцами кадра (x = 4.5, 5.5, 6.5, 7.5).
        FrameConvert.crop(кадр, Box(4.5f, 4f, 4f, 4f, 1f), pad = 0f, spec = спек, dst = тензор)
        val яркостьПосередине = 1.164f * ((16 + 5 * 8) + 0.5f * 8 - 16f) / 255f
        assertEquals(яркостьПосередине, тензор[3], 0.005f)
    }

    @Test
    fun `кроп берёт область бокса, а не весь кадр`() {
        val width = 32
        val height = 32
        val yРяд = width
        val yБуфер = ByteBuffer.allocate(yРяд * height)
        // Левая половина тёмная, правая светлая.
        for (строка in 0 until height) {
            for (столбец in 0 until width) {
                yБуфер.put(строка * yРяд + столбец, (if (столбец < width / 2) 20 else 200).toByte())
            }
        }
        val серый = ByteBuffer.allocate(width * (height / 2))
        for (индекс in 0 until серый.capacity()) серый.put(индекс, 128.toByte())
        val кадр = YuvFrame(
            width = width, height = height, rotationDegrees = 0,
            y = yБуфер, yRowStride = yРяд, yPixelStride = 1,
            u = серый, v = серый, uvRowStride = width, uvPixelStride = 1,
        )
        val спек = TensorSpec(
            size = 4, layout = TensorLayout.NHWC, channels = ChannelOrder.RGB,
            scaling = Scaling.Unit01,
        )
        val тензор = FloatArray(спек.floatCount)
        FrameConvert.crop(кадр, Box(18f, 4f, 12f, 12f, 0.9f), pad = 0f, spec = спек, dst = тензор)
        assertTrue("кроп правой половины обязан быть светлым", тензор.all { it > 0.6f })
    }

    @Test
    fun `поворот 90 берёт пиксели из своего угла сенсора`() {
        // Кадр 16×8, левая половина сенсора тёмная. При повороте на 90 деление кадра
        // едет с оси x на ось y: верх выпрямленного кадра — тёмный, низ — светлый.
        // Без поворота обе половины по y были бы одинаковыми, и тест это ловит.
        val width = 16
        val height = 8
        val yБуфер = ByteBuffer.allocate(width * height)
        for (строка in 0 until height) {
            for (столбец in 0 until width) {
                yБуфер.put(строка * width + столбец, (if (столбец < width / 2) 20 else 200).toByte())
            }
        }
        val серый = ByteBuffer.allocate(width * (height / 2))
        for (индекс in 0 until серый.capacity()) серый.put(индекс, 128.toByte())
        val кадр = YuvFrame(
            width = width, height = height, rotationDegrees = 90,
            y = yБуфер, yRowStride = width, yPixelStride = 1,
            u = серый, v = серый, uvRowStride = width, uvPixelStride = 1,
        )
        assertEquals(UprightSize(8, 16), кадр.upright)
        val спек = TensorSpec(
            size = 2, layout = TensorLayout.NHWC, channels = ChannelOrder.RGB,
            scaling = Scaling.Unit01,
        )
        val верх = FloatArray(спек.floatCount)
        FrameConvert.crop(кадр, Box(0f, 0f, 8f, 4f, 1f), pad = 0f, spec = спек, dst = верх)
        assertTrue("верх выпрямленного кадра обязан быть тёмным: ${верх.toList()}", верх.all { it < 0.2f })
        val низ = FloatArray(спек.floatCount)
        FrameConvert.crop(кадр, Box(0f, 12f, 8f, 4f, 1f), pad = 0f, spec = спек, dst = низ)
        assertTrue("низ выпрямленного кадра обязан быть светлым: ${низ.toList()}", низ.all { it > 0.6f })
    }

    @Test
    fun `короткий буфер назначения роняет`() {
        val кадр = ровныйКадр(16, 16, r = 10, g = 10, b = 10)
        val отказ = runCatching {
            FrameConvert.letterbox(кадр, спекДетектора, padValue = 114, dst = FloatArray(3))
        }.exceptionOrNull()
        assertTrue(отказ is IllegalArgumentException)
    }

    @Test
    fun `кадр короче своих шагов роняет с обоими числами`() {
        // Перепутанные плоскости и шаги — самая частая ошибка адаптера от `Image`:
        // без проверки она даёт IndexOutOfBounds из глубины выборки или чужие пиксели.
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            YuvFrame(
                width = 16, height = 16, rotationDegrees = 0,
                y = ByteBuffer.allocate(16 * 4), yRowStride = 16, yPixelStride = 1,
                u = ByteBuffer.allocate(8 * 8), v = ByteBuffer.allocate(8 * 8),
                uvRowStride = 8, uvPixelStride = 1,
            )
        }
        assertTrue(отказ.message!!.contains("64"))
        assertTrue(отказ.message!!.contains("256"))
    }

    @Test
    fun `mean и std не по три значения роняют`() {
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            Scaling.MeanStd(mean = floatArrayOf(0.5f, 0.5f), std = floatArrayOf(0.25f, 0.25f, 0.25f))
        }
        assertTrue(отказ.message!!.contains("2"))
    }
}
