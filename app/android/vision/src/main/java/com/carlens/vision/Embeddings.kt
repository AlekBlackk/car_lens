package com.carlens.vision

import java.security.MessageDigest
import kotlin.math.sqrt

/** Работа с эмбеддингами на границе плагина. */
object Embeddings {

    /**
     * Проверить внешний вектор и вернуть его нормализованную копию.
     *
     * Нормализация обязательна, потому что центроиды индекса нормализованы, а
     * приходящий снаружи вектор — нет. На M-1.6 сюда приходит EMA по треку, то есть
     * среднее нормированных векторов, а оно нормированным не является: его длина тем
     * меньше, чем сильнее расходились кадры. Скалярное произведение с центроидами
     * тогда систематически ниже косинуса, пороги §5.2 спеки плывут, а порядок классов
     * сохраняется — то есть тест на top-1 подмену не замечает.
     *
     * Копия, а не правка на месте: массив принадлежит вызывающему, и на M-1.6 это
     * буфер, который JS переиспользует между кадрами.
     */
    fun l2NormalizedCopy(embedding: FloatArray, expectedDim: Int): FloatArray {
        require(embedding.size == expectedDim) {
            "размерность вектора ${embedding.size} против $expectedDim у индекса"
        }
        var сумма = 0.0
        for (значение in embedding) {
            require(значение.isFinite()) { "в векторе неконечное значение: $значение" }
            сумма += (значение * значение).toDouble()
        }
        val норма = sqrt(сумма).toFloat()
        require(норма > 1e-12f) { "норма вектора $норма: нормализовать нечего" }
        val результат = FloatArray(embedding.size)
        for (индекс in embedding.indices) результат[индекс] = embedding[индекс] / норма
        return результат
    }

    fun sha256(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { байт ->
            "%02x".format(байт)
        }
}
