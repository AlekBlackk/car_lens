package com.carlens.vision

/**
 * Выборка пикселей источника кадра сразу в тензор модели.
 *
 * Источник — `FrameSource`: плоскости YUV с камеры или готовые RGB-пиксели в тестах
 * паритета. Геометрия, поля и раскладка живут здесь и общие для обоих, различается
 * только чтение цвета — иначе тестовый путь проверял бы свою копию арифметики.
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
    fun letterbox(frame: FrameSource, spec: TensorSpec, padValue: Int, dst: FloatArray): Letterbox {
        require(dst.size >= spec.floatCount) {
            "буфер назначения ${dst.size} против ${spec.floatCount} по спецификации тензора"
        }
        require(padValue in 0..255) { "значение поля $padValue вне 0..255" }
        val выпрямленный = frame.upright
        val отображение = Letterbox.forFrame(выпрямленный, spec.size)
        val цвет = FloatArray(3)
        for (строка in 0 until spec.size) {
            for (столбец in 0 until spec.size) {
                val xВыпрямленный = (столбец + 0.5f - отображение.padX) / отображение.scale - 0.5f
                val yВыпрямленный = (строка + 0.5f - отображение.padY) / отображение.scale - 0.5f
                записать(
                    frame, выпрямленный, spec, dst,
                    строка, столбец, xВыпрямленный, yВыпрямленный, padValue, цвет,
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
    fun crop(frame: FrameSource, boxUpright: Box, pad: Float, spec: TensorSpec, dst: FloatArray) {
        require(dst.size >= spec.floatCount) {
            "буфер назначения ${dst.size} против ${spec.floatCount} по спецификации тензора"
        }
        val выпрямленный = frame.upright
        val область = Boxes.pad(boxUpright, выпрямленный.width, выпрямленный.height, pad)
        require(область.w > 0f && область.h > 0f) { "пустой бокс: ${область.w}×${область.h}" }
        val цвет = FloatArray(3)
        for (строка in 0 until spec.size) {
            for (столбец in 0 until spec.size) {
                val xВыпрямленный = область.x + (столбец + 0.5f) * область.w / spec.size - 0.5f
                val yВыпрямленный = область.y + (строка + 0.5f) * область.h / spec.size - 0.5f
                записать(
                    frame, выпрямленный, spec, dst,
                    строка, столбец, xВыпрямленный, yВыпрямленный, БЕЗ_ПОЛЯ, цвет,
                )
            }
        }
    }

    private fun записать(
        frame: FrameSource,
        выпрямленный: UprightSize,
        spec: TensorSpec,
        dst: FloatArray,
        строка: Int,
        столбец: Int,
        xВыпрямленный: Float,
        yВыпрямленный: Float,
        padValue: Int,
        цвет: FloatArray,
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
        val xБуфера =
            Rotation.sensorX(xВыпрямленный, yВыпрямленный, выпрямленный, frame.rotationDegrees)
        val yБуфера =
            Rotation.sensorY(xВыпрямленный, yВыпрямленный, выпрямленный, frame.rotationDegrees)
        frame.sampleRgb(xБуфера, yБуфера, цвет)
        записатьПиксель(spec, dst, строка, столбец, цвет[0], цвет[1], цвет[2])
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

}
