package com.carlens.vision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

class EmbeddingsTest {

    @Test
    fun `нормализация даёт единичную длину`() {
        val вектор = floatArrayOf(3f, 4f, 0f, 0f)
        val результат = Embeddings.l2NormalizedCopy(вектор, expectedDim = 4)
        val длина = sqrt(результат.sumOf { (it * it).toDouble() }).toFloat()
        assertEquals(1f, длина, 1e-6f)
    }

    @Test
    fun `входной массив не изменяется`() {
        val вектор = floatArrayOf(3f, 4f, 0f, 0f)
        val копия = вектор.copyOf()
        val результат = Embeddings.l2NormalizedCopy(вектор, expectedDim = 4)
        assertArrayEquals(копия, вектор, 0f)
        assertNotSame(вектор, результат)
    }

    @Test
    fun `чужая размерность роняет с обоими числами`() {
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            Embeddings.l2NormalizedCopy(FloatArray(128), expectedDim = 256)
        }
        assertTrue(отказ.message!!.contains("128"))
        assertTrue(отказ.message!!.contains("256"))
    }

    @Test
    fun `nan в векторе роняет`() {
        val вектор = floatArrayOf(1f, Float.NaN, 0f, 0f)
        assertThrows(IllegalArgumentException::class.java) {
            Embeddings.l2NormalizedCopy(вектор, expectedDim = 4)
        }
    }

    @Test
    fun `бесконечность в векторе роняет`() {
        val вектор = floatArrayOf(1f, Float.POSITIVE_INFINITY, 0f, 0f)
        assertThrows(IllegalArgumentException::class.java) {
            Embeddings.l2NormalizedCopy(вектор, expectedDim = 4)
        }
    }

    @Test
    fun `нулевая норма роняет и называет норму`() {
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            Embeddings.l2NormalizedCopy(FloatArray(4), expectedDim = 4)
        }
        assertTrue(отказ.message!!.contains("0"))
    }
}
