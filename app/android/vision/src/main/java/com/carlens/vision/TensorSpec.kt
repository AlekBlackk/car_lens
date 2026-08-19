package com.carlens.vision

import java.nio.ByteBuffer

enum class TensorLayout { NHWC, NCHW }

enum class ChannelOrder { RGB, BGR }

sealed interface Scaling {
    /** Деление на 255 без вычитания среднего: так ждёт вход ultralytics. */
    data object Unit01 : Scaling

    /** Нормализация классификатора: значения из `export_report.json`. */
    data class MeanStd(val mean: FloatArray, val std: FloatArray) : Scaling {
        init {
            require(mean.size == 3 && std.size == 3) {
                "mean ${mean.size} и std ${std.size}: обоим полагается по 3 значения на канал"
            }
            require(std.all { it > 1e-12f }) { "нулевой std: ${std.toList()}" }
        }

        // `equals`/`hashCode` по содержимому: у data class с массивами они сравнивают
        // ссылки, а спецификация тензора сравнивается в тестах и в сверке контрактов.
        override fun equals(other: Any?): Boolean =
            this === other ||
                (other is MeanStd && mean.contentEquals(other.mean) && std.contentEquals(other.std))

        override fun hashCode(): Int = 31 * mean.contentHashCode() + std.contentHashCode()
    }
}

data class TensorSpec(
    val size: Int,
    val layout: TensorLayout,
    val channels: ChannelOrder,
    val scaling: Scaling,
) {
    init {
        require(size > 0) { "сторона тензора $size" }
    }

    val floatCount: Int get() = size * size * 3
}

/**
 * Кадр YUV_420_888 как набор буферов и шагов. Отдельный тип, а не `android.media.Image`:
 * так вся арифметика препроцессинга проверяется JVM-тестами без устройства, а адаптер
 * от `Image` остаётся тонким и проверяется на телефоне.
 */
class YuvFrame(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val y: ByteBuffer,
    val yRowStride: Int,
    val yPixelStride: Int,
    val u: ByteBuffer,
    val v: ByteBuffer,
    val uvRowStride: Int,
    val uvPixelStride: Int,
) {
    /** Ширина плоскости хромы: у YUV_420_888 она вдвое меньше, с округлением вверх. */
    val uvWidth: Int get() = (width + 1) / 2

    val uvHeight: Int get() = (height + 1) / 2

    val upright: UprightSize get() = Rotation.upright(width, height, rotationDegrees)

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
