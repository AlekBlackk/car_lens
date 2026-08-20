package com.carlens.vision

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class DelegateMode { GPU, CPU }

/**
 * Разбор выхода графа в рамки кадра — отдельно от рантайма.
 *
 * Вынесено из `Detector`, потому что это чистая арифметика и самая ошибкоопасная часть
 * задачи: индексация `CN` против `NC`, фильтр по классам COCO и порядок «NMS, потом
 * разворот». Внутри `Detector` она проверялась бы только на телефоне, а так — обычными
 * JVM-тестами в каждом прогоне CI.
 */
internal object DetectorDecode {

    fun boxes(
        raw: FloatArray,
        contract: DetectorContract,
        каналов: Int,
        якорей: Int,
        единицы: Float,
        отображение: Letterbox,
    ): List<Box> {
        val кандидаты = ArrayList<Box>()
        for (якорь in 0 until якорей) {
            var лучший = 0f
            for (класс in contract.vehicleClasses) {
                val скор = значение(raw, contract.outputLayout, каналов, якорей, 4 + класс, якорь)
                if (скор > лучший) лучший = скор
            }
            if (лучший < ПОРОГ_УВЕРЕННОСТИ) continue
            кандидаты.add(
                Boxes.fromRaw(
                    cx = значение(raw, contract.outputLayout, каналов, якорей, 0, якорь),
                    cy = значение(raw, contract.outputLayout, каналов, якорей, 1, якорь),
                    a = значение(raw, contract.outputLayout, каналов, якорей, 2, якорь),
                    b = значение(raw, contract.outputLayout, каналов, якорей, 3, якорь),
                    format = contract.boxFormat,
                    unitsScale = единицы,
                    conf = лучший,
                )
            )
        }
        if (кандидаты.isEmpty()) return emptyList()
        // Порядок обязателен: NMS считается в координатах тензора, где боксы сравнимы
        // между собой, и только выжившие разворачиваются обратно в кадр.
        return Boxes.nms(кандидаты, ПОРОГ_IOU, contract.maxDet)
            .map { отображение.tensorToUpright(it) }
    }

    private fun значение(
        raw: FloatArray,
        раскладка: String,
        каналов: Int,
        якорей: Int,
        канал: Int,
        якорь: Int,
    ): Float = when (раскладка) {
        "CN" -> raw[канал * якорей + якорь]
        else -> raw[якорь * каналов + канал]
    }
}

/** Порог уверенности и IoU для NMS — §4.1 спеки, те же числа, что у питоновской стороны. */
private const val ПОРОГ_УВЕРЕННОСТИ = 0.35f
private const val ПОРОГ_IOU = 0.5f

/**
 * Детектор транспорта поверх экспортированного графа.
 *
 * Каждое поле контракта манифеста сверяется с графом при загрузке: контракт пишет
 * питоновская сторона, граф собирает конвертер, и разошедшаяся пара даёт не отказ, а
 * правдоподобные рамки не на машинах.
 */
internal class Detector(
    model: ByteBuffer,
    private val contract: DetectorContract,
    spec: TensorSpec,
    allowGpu: Boolean = true,
) : AutoCloseable {

    private val поднятый: Delegates.Поднятый
    private val интерпретатор: Interpreter
    val delegateMode: DelegateMode

    private val якорей: Int
    private val каналов: Int
    private val спек: TensorSpec
    private val пиксели: FloatArray
    private val входнойБуфер: ByteBuffer
    private val выходнойБуфер: ByteBuffer
    private val сырыеВыходы: FloatArray
    private val единицы: Float

    init {
        поднятый = Delegates.поднять(model, allowGpu, cpuThreads = ЧИСЛО_ПОТОКОВ)
        интерпретатор = поднятый.interpreter
        delegateMode = поднятый.mode

        val вход = интерпретатор.getInputTensor(0)
        check(вход.dataType() == DataType.FLOAT32) {
            "тип входа детектора ${вход.dataType()} против float32 в контракте"
        }
        val формаВхода = вход.shape()
        check(формаВхода.size == 4 && формаВхода[0] == 1) {
            "форма входа детектора ${формаВхода.toList()}, ожидались четыре оси с единицей в первой"
        }
        val каналыВпереди = формаВхода[1] == 3
        val сторона = if (каналыВпереди) формаВхода[2] else формаВхода[1]
        check(сторона == contract.imgsz) {
            "сторона входа графа $сторона против ${contract.imgsz} в контракте"
        }
        val раскладкаГрафа = if (каналыВпереди) TensorLayout.NCHW else TensorLayout.NHWC
        check(раскладкаГрафа == spec.layout) {
            "раскладка входа графа $раскладкаГрафа против ${spec.layout} в контракте"
        }
        спек = spec

        val формаВыхода = интерпретатор.getOutputTensor(0).shape()
        check(формаВыхода.size == 3 && формаВыхода[0] == 1) {
            "форма выхода детектора ${формаВыхода.toList()}, ожидалось три оси с единицей в первой"
        }
        when (contract.outputLayout) {
            "CN" -> {
                каналов = формаВыхода[1]
                якорей = формаВыхода[2]
            }
            "NC" -> {
                якорей = формаВыхода[1]
                каналов = формаВыхода[2]
            }
            else -> throw IllegalStateException(
                "неизвестный output_layout \"${contract.outputLayout}\""
            )
        }
        check(каналов == 4 + contract.numClasses) {
            "каналов выхода $каналов против ${4 + contract.numClasses} по контракту " +
                "(4 координаты и ${contract.numClasses} классов)"
        }
        check(contract.scoreLayout == "classes_only") {
            "score_layout \"${contract.scoreLayout}\" не поддержан: веха ждёт classes_only"
        }
        единицы = when (contract.coordUnits) {
            "normalized" -> contract.imgsz.toFloat()
            "pixels" -> 1f
            else -> throw IllegalStateException(
                "неизвестный coord_units \"${contract.coordUnits}\""
            )
        }

        пиксели = FloatArray(спек.floatCount)
        входнойБуфер = ByteBuffer.allocateDirect(вход.numBytes()).order(ByteOrder.nativeOrder())
        выходнойБуфер = ByteBuffer
            .allocateDirect(интерпретатор.getOutputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())
        сырыеВыходы = FloatArray(каналов * якорей)
    }

    /**
     * Рамки транспорта в координатах выпрямленного кадра.
     *
     * GPU-делегат может не подняться, и тогда всё работает медленнее, но одинаково: это
     * видно в `delegateMode`, а не в тишине.
     */
    fun detect(frame: FrameSource): List<Box> {
        val отображение = FrameConvert.letterbox(
            frame, спек, contract.letterboxPadValue, пиксели,
        )
        входнойБуфер.rewind()
        for (значение in пиксели) входнойБуфер.putFloat(значение)
        входнойБуфер.rewind()
        выходнойБуфер.rewind()
        интерпретатор.run(входнойБуфер, выходнойБуфер)
        выходнойБуфер.rewind()
        for (индекс in сырыеВыходы.indices) сырыеВыходы[индекс] = выходнойБуфер.getFloat()
        return DetectorDecode.boxes(сырыеВыходы, contract, каналов, якорей, единицы, отображение)
    }

    override fun close() = поднятый.close()

    private companion object {
        const val ЧИСЛО_ПОТОКОВ = 4
    }
}
