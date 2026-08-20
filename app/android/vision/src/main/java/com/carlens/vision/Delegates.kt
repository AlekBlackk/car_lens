package com.carlens.vision

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer

/**
 * Подъём интерпретатора с попыткой GPU-делегата и откатом на CPU.
 *
 * Общее место для детектора и классификатора: оба поднимаются одинаково — GPU, если
 * разрешён и поддержан устройством, иначе CPU в несколько потоков, — и раздельные копии
 * этой логики разошлись бы при первой правке одной из них. Отказ GPU не фатален: CPU-путь
 * даёт те же числа, только медленнее. Молчать о подмене нельзя — она объясняет и время
 * кадра, и разряд батареи, поэтому режим уезжает наружу в `DelegateMode`, а не тонет здесь.
 */
internal object Delegates {

    class Поднятый(val interpreter: Interpreter, val mode: DelegateMode, private val gpu: GpuDelegate?) {
        fun close() {
            interpreter.close()
            gpu?.close()
        }
    }

    fun поднять(model: ByteBuffer, allowGpu: Boolean, cpuThreads: Int): Поднятый {
        val список = CompatibilityList()
        if (allowGpu && список.use { it.isDelegateSupportedOnThisDevice }) {
            val делегатGpu = GpuDelegate()
            val результат = runCatching {
                Interpreter(model, Interpreter.Options().addDelegate(делегатGpu))
            }
            результат.getOrNull()?.let { return Поднятый(it, DelegateMode.GPU, делегатGpu) }
            делегатGpu.close()
        }
        model.rewind()
        val параметры = Interpreter.Options().setNumThreads(cpuThreads)
        return Поднятый(Interpreter(model, параметры), DelegateMode.CPU, null)
    }
}
