package com.carlens.vision

import java.nio.ByteBuffer
import kotlin.math.floor
import kotlin.math.min

/**
 * Источник пикселей для препроцессинга.
 *
 * Отдаёт цвет по дробным координатам своего буфера — всё остальное (letterbox, кроп,
 * поворот, нормализация, раскладка) живёт в `FrameConvert` и одинаково для любого
 * источника. Реализаций две: кадр камеры `YuvFrame` и `RgbFrame` для тестов паритета,
 * где вход приходит уже готовыми пикселями PNG.
 */
sealed interface FrameSource {

    val width: Int
    val height: Int
    val rotationDegrees: Int

    val upright: UprightSize get() = Rotation.upright(width, height, rotationDegrees)

    /**
     * Билинейно выбрать цвет в координатах буфера и положить в `out` как R, G, B в
     * диапазоне 0..255. Массив передаётся снаружи: на кадр 416×416 это 173 тысячи
     * вызовов, и своя аллокация в каждом свела бы на нет отказ от промежуточного битмапа.
     */
    fun sampleRgb(x: Float, y: Float, out: FloatArray)
}

/**
 * Кадр YUV_420_888 как набор буферов и шагов. Отдельный тип, а не `android.media.Image`:
 * так вся арифметика препроцессинга проверяется JVM-тестами без устройства, а адаптер
 * от `Image` остаётся тонким и проверяется на телефоне.
 */
class YuvFrame(
    override val width: Int,
    override val height: Int,
    override val rotationDegrees: Int,
    val y: ByteBuffer,
    val yRowStride: Int,
    val yPixelStride: Int,
    val u: ByteBuffer,
    val v: ByteBuffer,
    val uvRowStride: Int,
    val uvPixelStride: Int,
) : FrameSource {

    /** Ширина плоскости хромы: у YUV_420_888 она вдвое меньше, с округлением вверх. */
    val uvWidth: Int get() = (width + 1) / 2

    val uvHeight: Int get() = (height + 1) / 2

    init {
        require(width > 0 && height > 0) { "пустой кадр ${width}x$height" }
        require(yRowStride > 0 && yPixelStride > 0) {
            "шаги плоскости Y: ряд $yRowStride, пиксель $yPixelStride"
        }
        require(uvRowStride > 0 && uvPixelStride > 0) {
            "шаги плоскостей хромы: ряд $uvRowStride, пиксель $uvPixelStride"
        }
        // Перепутанные местами плоскости или шаги — самая частая ошибка адаптера от
        // `Image`, и без этой сверки она даёт либо чужие пиксели, либо
        // IndexOutOfBounds из глубины выборки, где по номеру байта уже ничего не понять.
        сверитьВместимость("Y", y, yRowStride, yPixelStride, width, height)
        сверитьВместимость("U", u, uvRowStride, uvPixelStride, uvWidth, uvHeight)
        сверитьВместимость("V", v, uvRowStride, uvPixelStride, uvWidth, uvHeight)
    }

    override fun sampleRgb(x: Float, y: Float, out: FloatArray) {
        val яркость = Bilinear.выбрать(
            this.y, yRowStride, yPixelStride, width, height, x, y,
        )
        // Пиксель хромы `j` покрывает пиксели яркости `2j` и `2j+1`, то есть его центр
        // лежит в `2j + 0.5` координат сенсора — отсюда обратный перевод.
        val xХромы = (x - 0.5f) / 2f
        val yХромы = (y - 0.5f) / 2f
        val синева = Bilinear.выбрать(
            u, uvRowStride, uvPixelStride, uvWidth, uvHeight, xХромы, yХромы,
        )
        val краснота = Bilinear.выбрать(
            v, uvRowStride, uvPixelStride, uvWidth, uvHeight, xХромы, yХромы,
        )
        // BT.601, ограниченный диапазон: Y 16…235, U/V 16…240 — так отдаёт камера.
        val линейная = 1.164f * (яркость - 16f)
        val дельтаU = синева - 128f
        val дельтаV = краснота - 128f
        out[0] = (линейная + 1.596f * дельтаV).coerceIn(0f, 255f)
        out[1] = (линейная - 0.813f * дельтаV - 0.391f * дельтаU).coerceIn(0f, 255f)
        out[2] = (линейная + 2.018f * дельтаU).coerceIn(0f, 255f)
    }

    private fun сверитьВместимость(
        имя: String,
        буфер: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        ширина: Int,
        высота: Int,
    ) {
        val нужно = rowStride * (высота - 1) + (ширина - 1) * pixelStride + 1
        require(буфер.capacity() >= нужно) {
            "плоскость $имя: ${буфер.capacity()} байт против $нужно по шагам ${ширина}x$высота"
        }
    }
}

/**
 * Кадр из готовых пикселей ARGB — вход тестов паритета.
 *
 * Конвертации YUV→RGB у питоновского эталона нет, и сверять её не с чем (§6.4 дизайна):
 * эталонные картинки приходят PNG, то есть уже RGB. Всё остальное — та же геометрия и
 * та же нормализация, что у кадра камеры.
 */
class RgbFrame(
    private val pixels: IntArray,
    override val width: Int,
    override val height: Int,
    override val rotationDegrees: Int = 0,
) : FrameSource {

    init {
        require(width > 0 && height > 0) { "пустой кадр ${width}x$height" }
        require(pixels.size >= width * height) {
            "пикселей ${pixels.size} против ${width * height} по размеру кадра"
        }
    }

    override fun sampleRgb(x: Float, y: Float, out: FloatArray) {
        Bilinear.выбратьARGB(pixels, width, height, x, y, out)
    }
}

/** Билинейная выборка с зажатием координат по границам буфера. */
internal object Bilinear {

    fun выбрать(
        буфер: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        ширина: Int,
        высота: Int,
        x: Float,
        y: Float,
    ): Float {
        val xЗажатый = x.coerceIn(0f, ширина - 1f)
        val yЗажатый = y.coerceIn(0f, высота - 1f)
        val x0 = floor(xЗажатый).toInt()
        val y0 = floor(yЗажатый).toInt()
        val x1 = min(x0 + 1, ширина - 1)
        val y1 = min(y0 + 1, высота - 1)
        val долиX = xЗажатый - x0
        val долиY = yЗажатый - y0
        val верхнийРяд = y0 * rowStride
        val нижнийРяд = y1 * rowStride
        val верх = смешать(
            байт(буфер, верхнийРяд + x0 * pixelStride),
            байт(буфер, верхнийРяд + x1 * pixelStride),
            долиX,
        )
        val низ = смешать(
            байт(буфер, нижнийРяд + x0 * pixelStride),
            байт(буфер, нижнийРяд + x1 * pixelStride),
            долиX,
        )
        return смешать(верх, низ, долиY)
    }

    fun выбратьARGB(
        pixels: IntArray,
        ширина: Int,
        высота: Int,
        x: Float,
        y: Float,
        out: FloatArray,
    ) {
        val xЗажатый = x.coerceIn(0f, ширина - 1f)
        val yЗажатый = y.coerceIn(0f, высота - 1f)
        val x0 = floor(xЗажатый).toInt()
        val y0 = floor(yЗажатый).toInt()
        val x1 = min(x0 + 1, ширина - 1)
        val y1 = min(y0 + 1, высота - 1)
        val долиX = xЗажатый - x0
        val долиY = yЗажатый - y0
        val слеваСверху = pixels[y0 * ширина + x0]
        val справаСверху = pixels[y0 * ширина + x1]
        val слеваСнизу = pixels[y1 * ширина + x0]
        val справаСнизу = pixels[y1 * ширина + x1]
        for (канал in 0 until 3) {
            val сдвиг = 16 - канал * 8 // ARGB: красный 16, зелёный 8, синий 0
            val верх = смешать(
                ((слеваСверху shr сдвиг) and 0xFF).toFloat(),
                ((справаСверху shr сдвиг) and 0xFF).toFloat(),
                долиX,
            )
            val низ = смешать(
                ((слеваСнизу shr сдвиг) and 0xFF).toFloat(),
                ((справаСнизу shr сдвиг) and 0xFF).toFloat(),
                долиX,
            )
            out[канал] = смешать(верх, низ, долиY)
        }
    }

    private fun смешать(первое: Float, второе: Float, доля: Float): Float =
        первое + (второе - первое) * доля

    private fun байт(буфер: ByteBuffer, индекс: Int): Float =
        (буфер.get(индекс).toInt() and 0xFF).toFloat()
}
