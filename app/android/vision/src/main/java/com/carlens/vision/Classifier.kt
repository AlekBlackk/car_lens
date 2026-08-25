package com.carlens.vision

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
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

    /*
     * Пакетный обмен с прямыми буферами, та же причина, что у `Detector`: поэлементные
     * `putFloat`/`getFloat`/`put`/`get` — вызов с проверкой границ на каждое из 307 200
     * значений входа 320x320x3, тогда как `put(FloatArray)`/`put(ByteArray)` на буфере в
     * порядке байтов платформы сводятся к `memcpy`. Окна и промежуточный `ByteArray`
     * заводятся один раз: аллокация в горячем пути видна рывками, а не замедлением.
     */
    private val входныеЧисла: FloatBuffer
    private val выходныеЧисла: FloatBuffer
    private val входныеБайты: ByteArray
    private val выходныеБайты: ByteArray

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
        входныеЧисла = входнойБуфер.asFloatBuffer()
        выходныеЧисла = выходнойБуфер.asFloatBuffer()
        входныеБайты = ByteArray(вход.numBytes())
        выходныеБайты = ByteArray(выход.numBytes())
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
            DataType.FLOAT32 -> {
                входныеЧисла.clear()
                входныеЧисла.put(пиксели, 0, tensorSpec.floatCount)
            }
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
        // Квантование остаётся поэлементным — оно и есть работа; пакетным делается только
        // перенос в прямой буфер, где поэлементный `put` был чистыми накладными расходами.
        for (индекс in пиксели.indices) {
            val квант = (пиксели[индекс] / параметры.scale).roundToInt() + параметры.zeroPoint
            входныеБайты[индекс] = квант.coerceIn(диапазон.first, диапазон.last).toByte()
        }
        входнойБуфер.put(входныеБайты, 0, пиксели.size)
    }

    private fun прочитатьВыход(): FloatArray {
        выходнойБуфер.rewind()
        val результат = FloatArray(embeddingDim)
        when (выход.dataType()) {
            DataType.FLOAT32 -> {
                выходныеЧисла.clear()
                выходныеЧисла.get(результат, 0, embeddingDim)
            }
            DataType.UINT8, DataType.INT8 -> {
                val параметры = выход.quantizationParams()
                val знаковый = выход.dataType() == DataType.INT8
                выходнойБуфер.get(выходныеБайты, 0, embeddingDim)
                for (индекс in результат.indices) {
                    val байт = выходныеБайты[индекс].toInt()
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
        /**
         * Столько же, сколько у детектора, а не вдвое меньше.
         *
         * Двойка стояла здесь с M-1.5 без обоснования в коде, а стадия стоит 47 мс при
         * бюджете §5.4 в 10 (`bench/latency.md`) и делит устройство с детектором, который
         * считается на GPU, то есть больших ядер не занимает. Ускорение ожидается от
         * распараллеливания графа, а не от смены математики: TFLite делит между потоками
         * выход операции, и порядок накопления внутри одного выходного элемента от числа
         * потоков не зависит.
         *
         * «Ожидается» — потому что порог приёмки тут жёсткий, а рассуждение прогоном не
         * является: паритет уровня A требует косинуса ≥0.9999 с питоновским эталоном, и
         * ровно на нём сорвался GPU-делегат (см. `bench/latency.md`). Правку обязан
         * подтвердить `:vision:connectedReleaseAndroidTest` — `ParityTest` вместе с
         * `LatencyTest`, иначе её надо откатить, а не «оставить, вроде работает».
         */
        const val ЧИСЛО_ПОТОКОВ = 4
    }
}
