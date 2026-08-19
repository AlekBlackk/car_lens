package com.carlens.vision

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
