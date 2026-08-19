package com.carlens.vision

import java.nio.ByteBuffer
import kotlin.math.floor
import kotlin.math.min

/**
 * Выборка пикселей из плоскостей YUV_420_888 сразу в тензор модели.
 *
 * Полный RGB-кадр не материализуется: промежуточный битмап — это лишний проход по
 * памяти на каждом кадре, и ровно на его отсутствии стоит отказ от NDK (§3 дизайна).
 * Выборка билинейная, потому что питоновский эталон и в ultralytics, и в `A.Resize`
 * валидации использует `cv2.INTER_LINEAR`; ближайший сосед сдвигает эмбеддинг ровно
 * настолько, чтобы паритет не сошёлся, а причину искали в квантизации.
 */
object FrameConvert {

    private const val БЕЗ_ПОЛЯ = -1

    /**
     * Вписать кадр целиком в квадратный тензор с серыми полями.
     *
     * Возвращает то же отображение, что `Letterbox.forFrame`: вызывающий разворачивает
     * им боксы обратно в координаты выпрямленного кадра.
     */
    fun letterbox(frame: YuvFrame, spec: TensorSpec, padValue: Int, dst: FloatArray): Letterbox {
        require(dst.size >= spec.floatCount) {
            "буфер назначения ${dst.size} против ${spec.floatCount} по спецификации тензора"
        }
        require(padValue in 0..255) { "значение поля $padValue вне 0..255" }
        val выпрямленный = frame.upright
        val отображение = Letterbox.forFrame(выпрямленный, spec.size)
        for (строка in 0 until spec.size) {
            for (столбец in 0 until spec.size) {
                val xВыпрямленный = (столбец + 0.5f - отображение.padX) / отображение.scale - 0.5f
                val yВыпрямленный = (строка + 0.5f - отображение.padY) / отображение.scale - 0.5f
                записать(
                    frame, выпрямленный, spec, dst,
                    строка, столбец, xВыпрямленный, yВыпрямленный, padValue,
                )
            }
        }
        return отображение
    }

    /**
     * Растянуть область бокса на весь тензор — вход классификатора.
     *
     * Полей здесь нет: `Boxes.pad` уже обрезал область по кадру, а доли пикселя на краю
     * берутся зажатием координаты, а не серым, иначе кроп у края кадра получает рамку,
     * которой не было в обучающих кропах.
     */
    fun crop(frame: YuvFrame, boxUpright: Box, pad: Float, spec: TensorSpec, dst: FloatArray) {
        require(dst.size >= spec.floatCount) {
            "буфер назначения ${dst.size} против ${spec.floatCount} по спецификации тензора"
        }
        val выпрямленный = frame.upright
        val область = Boxes.pad(boxUpright, выпрямленный.width, выпрямленный.height, pad)
        require(область.w > 0f && область.h > 0f) { "пустой бокс: ${область.w}×${область.h}" }
        for (строка in 0 until spec.size) {
            for (столбец in 0 until spec.size) {
                val xВыпрямленный = область.x + (столбец + 0.5f) * область.w / spec.size - 0.5f
                val yВыпрямленный = область.y + (строка + 0.5f) * область.h / spec.size - 0.5f
                записать(
                    frame, выпрямленный, spec, dst,
                    строка, столбец, xВыпрямленный, yВыпрямленный, БЕЗ_ПОЛЯ,
                )
            }
        }
    }

    private fun записать(
        frame: YuvFrame,
        выпрямленный: UprightSize,
        spec: TensorSpec,
        dst: FloatArray,
        строка: Int,
        столбец: Int,
        xВыпрямленный: Float,
        yВыпрямленный: Float,
        padValue: Int,
    ) {
        val внеКадра = xВыпрямленный < 0f || yВыпрямленный < 0f ||
            xВыпрямленный > выпрямленный.width - 1f || yВыпрямленный > выпрямленный.height - 1f
        if (внеКадра && padValue != БЕЗ_ПОЛЯ) {
            // Поле проходит ту же нормализацию, что и пиксели: для детектора это
            // 114/255 из ultralytics, а серый, не поделённый на 255, был бы ровно тем
            // тихим расхождением с эталоном, которое ищут потом в квантизации.
            val серый = padValue.toFloat()
            записатьПиксель(spec, dst, строка, столбец, серый, серый, серый)
            return
        }
        val xСенсора =
            Rotation.sensorX(xВыпрямленный, yВыпрямленный, выпрямленный, frame.rotationDegrees)
        val yСенсора =
            Rotation.sensorY(xВыпрямленный, yВыпрямленный, выпрямленный, frame.rotationDegrees)
        val яркость = выбрать(
            frame.y, frame.yRowStride, frame.yPixelStride, frame.width, frame.height,
            xСенсора, yСенсора,
        )
        // Пиксель хромы `j` покрывает пиксели яркости `2j` и `2j+1`, то есть его центр
        // лежит в `2j + 0.5` координат сенсора — отсюда обратный перевод.
        val xХромы = (xСенсора - 0.5f) / 2f
        val yХромы = (yСенсора - 0.5f) / 2f
        val синева = выбрать(
            frame.u, frame.uvRowStride, frame.uvPixelStride, frame.uvWidth, frame.uvHeight,
            xХромы, yХромы,
        )
        val краснота = выбрать(
            frame.v, frame.uvRowStride, frame.uvPixelStride, frame.uvWidth, frame.uvHeight,
            xХромы, yХромы,
        )
        // BT.601, ограниченный диапазон: Y 16…235, U/V 16…240 — так отдаёт камера.
        val линейная = 1.164f * (яркость - 16f)
        val дельтаU = синева - 128f
        val дельтаV = краснота - 128f
        записатьПиксель(
            spec, dst, строка, столбец,
            (линейная + 1.596f * дельтаV).coerceIn(0f, 255f),
            (линейная - 0.813f * дельтаV - 0.391f * дельтаU).coerceIn(0f, 255f),
            (линейная + 2.018f * дельтаU).coerceIn(0f, 255f),
        )
    }

    private fun записатьПиксель(
        spec: TensorSpec,
        dst: FloatArray,
        строка: Int,
        столбец: Int,
        красный: Float,
        зелёный: Float,
        синий: Float,
    ) {
        val первый = if (spec.channels == ChannelOrder.RGB) красный else синий
        val третий = if (spec.channels == ChannelOrder.RGB) синий else красный
        положить(spec, dst, строка, столбец, канал = 0, значение = первый)
        положить(spec, dst, строка, столбец, канал = 1, значение = зелёный)
        положить(spec, dst, строка, столбец, канал = 2, значение = третий)
    }

    /**
     * Нормализовать и положить одно значение.
     *
     * `mean`/`std` индексируются номером канала **в тензоре**, а не цветом: в
     * `export_report.json` они записаны в том порядке, в каком модель ждёт входы, и
     * `ChannelOrder` уже применён выше.
     */
    private fun положить(
        spec: TensorSpec,
        dst: FloatArray,
        строка: Int,
        столбец: Int,
        канал: Int,
        значение: Float,
    ) {
        val нормированное = when (val масштаб = spec.scaling) {
            is Scaling.Unit01 -> значение / 255f
            is Scaling.MeanStd -> (значение / 255f - масштаб.mean[канал]) / масштаб.std[канал]
        }
        val смещение = when (spec.layout) {
            TensorLayout.NHWC -> (строка * spec.size + столбец) * 3 + канал
            TensorLayout.NCHW -> канал * spec.size * spec.size + строка * spec.size + столбец
        }
        dst[смещение] = нормированное
    }

    /** Билинейная выборка одного байта плоскости с зажатием координат по её границам. */
    private fun выбрать(
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
        val слева = y0 * rowStride
        val снизу = y1 * rowStride
        val верх = смешать(байт(буфер, слева + x0 * pixelStride), байт(буфер, слева + x1 * pixelStride), долиX)
        val низ = смешать(байт(буфер, снизу + x0 * pixelStride), байт(буфер, снизу + x1 * pixelStride), долиX)
        return смешать(верх, низ, долиY)
    }

    private fun смешать(первое: Float, второе: Float, доля: Float): Float =
        первое + (второе - первое) * доля

    private fun байт(буфер: ByteBuffer, индекс: Int): Float =
        (буфер.get(индекс).toInt() and 0xFF).toFloat()
}
