package com.carlens.vision

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Классификатор кузова: кроп бокса — эмбеддинг.
 *
 * Границы графа читаются из интерпретатора, а не предполагаются: раскладку входа
 * выбирает конвертер (`export/to_tflite.py:232`), а INT8-модель приходит с квантованной
 * границей. Всё, что можно сверить при загрузке, сверяется при загрузке: расхождение
 * формы выхода с индексом на первом кадре выглядело бы как плохое качество узнавания.
 */
internal class Classifier(
    model: ByteBuffer,
    spec: TensorSpec,
    expectedEmbeddingDim: Int,
) : AutoCloseable {

    // GPU принудительно выключен, а не читается из AssetSpec.allowGpu: замер на S23 Ultra
    // дал реальное ускорение (123 → 77 мс), но косинус паритета уровня A упал до 0.964
    // против требуемых 0.9999 (тест эталона, кадр 0). GPU-делегат на квантованном INT8-
    // графе где-то теряет точность — вероятно, часть операций считается в пониженной
    // точности вместо квантованной арифметики. Веха проверяет именно совпадение с
    // питоновским эталоном, поэтому скорость здесь не аргумент.
    private val поднятый = Delegates.поднять(model, allowGpu = false, cpuThreads = ЧИСЛО_ПОТОКОВ)
    private val интерпретатор = поднятый.interpreter
    val delegateMode: DelegateMode = поднятый.mode
    private val вход = интерпретатор.getInputTensor(0)
    private val выход = интерпретатор.getOutputTensor(0)

    /** Спецификация с раскладкой из графа, а не из манифеста: в манифесте её нет. */
    val tensorSpec: TensorSpec

    val embeddingDim: Int

    private val пиксели: FloatArray
    private val входнойБуфер: ByteBuffer
    private val выходнойБуфер: ByteBuffer

    init {
        val формаВхода = вход.shape()
        require(формаВхода.size == 4 && формаВхода[0] == 1) {
            "форма входа классификатора ${формаВхода.toList()}, ожидались четыре оси с единицей в первой"
        }
        val каналыВпереди = формаВхода[1] == 3
        val сторона = if (каналыВпереди) формаВхода[2] else формаВхода[1]
        val каналов = if (каналыВпереди) формаВхода[1] else формаВхода[3]
        check(каналов == 3) { "каналов на входе классификатора $каналов, а не 3" }
        check(сторона == spec.size) {
            "сторона входа графа $сторона против ${spec.size} в манифесте"
        }
        tensorSpec = spec.copy(
            layout = if (каналыВпереди) TensorLayout.NCHW else TensorLayout.NHWC,
        )

        val формаВыхода = выход.shape()
        check(формаВыхода.size == 2 && формаВыхода[0] == 1) {
            "форма выхода классификатора ${формаВыхода.toList()}, ожидалось [1, dim]"
        }
        embeddingDim = формаВыхода[1]
        check(embeddingDim == expectedEmbeddingDim) {
            "embedding_dim графа $embeddingDim против $expectedEmbeddingDim в индексе"
        }

        // Буферы аллоцируются один раз: на каждый кадр их может быть несколько (бокс на
        // машину), и мусор здесь виден как рывки, а не как замедление.
        пиксели = FloatArray(tensorSpec.floatCount)
        входнойБуфер = ByteBuffer.allocateDirect(вход.numBytes()).order(ByteOrder.nativeOrder())
        выходнойБуфер = ByteBuffer.allocateDirect(выход.numBytes()).order(ByteOrder.nativeOrder())
    }

    /** Эмбеддинг кропа, нормализованный L2. Массив свежий: он уходит наружу. */
    fun embed(frame: FrameSource, box: Box, pad: Float): FloatArray {
        FrameConvert.crop(frame, box, pad, tensorSpec, пиксели)
        записатьВход()
        выходнойБуфер.rewind()
        интерпретатор.run(входнойБуфер, выходнойБуфер)
        val сырой = прочитатьВыход()
        // Модель отдаёт ненормированный вектор (`training/model.py:81`), а центроиды
        // индекса нормализованы: без нормализации скалярное произведение — не косинус.
        return Embeddings.l2NormalizedCopy(сырой, embeddingDim)
    }

    private fun записатьВход() {
        входнойБуфер.rewind()
        when (вход.dataType()) {
            DataType.FLOAT32 -> for (значение in пиксели) входнойБуфер.putFloat(значение)
            DataType.UINT8 -> квантовать(диапазон = 0..255)
            DataType.INT8 -> квантовать(диапазон = -128..127)
            else -> throw IllegalStateException(
                "тип входа классификатора ${вход.dataType()} не поддержан"
            )
        }
        входнойБуфер.rewind()
    }

    private fun квантовать(диапазон: IntRange) {
        val параметры = вход.quantizationParams()
        check(параметры.scale > 0f) {
            "scale входа ${параметры.scale}: квантованная граница без масштаба"
        }
        for (значение in пиксели) {
            val квант = (значение / параметры.scale).roundToInt() + параметры.zeroPoint
            входнойБуфер.put(квант.coerceIn(диапазон.first, диапазон.last).toByte())
        }
    }

    private fun прочитатьВыход(): FloatArray {
        выходнойБуфер.rewind()
        val результат = FloatArray(embeddingDim)
        when (выход.dataType()) {
            DataType.FLOAT32 -> for (индекс in результат.indices) {
                результат[индекс] = выходнойБуфер.getFloat()
            }
            DataType.UINT8, DataType.INT8 -> {
                val параметры = выход.quantizationParams()
                val знаковый = выход.dataType() == DataType.INT8
                for (индекс in результат.indices) {
                    val байт = выходнойБуфер.get().toInt()
                    val квант = if (знаковый) байт else байт and 0xFF
                    результат[индекс] = (квант - параметры.zeroPoint) * параметры.scale
                }
            }
            else -> throw IllegalStateException(
                "тип выхода классификатора ${выход.dataType()} не поддержан"
            )
        }
        return результат
    }

    override fun close() = поднятый.close()

    private companion object {
        const val ЧИСЛО_ПОТОКОВ = 2
    }
}
