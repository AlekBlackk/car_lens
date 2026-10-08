package com.carlens.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertSame
import org.junit.Test

class BufferPoolTest {

    private class Буфер(val байт: Int)

    @Test
    fun `тот же размер даёт тот же объект`() {
        val пул = BufferPool { Буфер(it) }
        val первый = пул.взять(20)
        assertSame(первый, пул.взять(20))
        assertEquals(20, первый.байт)
    }

    @Test
    fun `разные размеры дают разные объекты точного размера`() {
        val пул = BufferPool { Буфер(it) }
        val малый = пул.взять(20)
        val большой = пул.взять(40)
        assertNotSame(малый, большой)
        assertEquals(20, малый.байт)
        assertEquals(40, большой.байт)
    }

    @Test
    fun `миллион кадров не создают больше буферов, чем размеров`() {
        // Ровно то, чего не было в плагине детекции: новый массив на каждый кадр.
        var выделено = 0
        val пул = BufferPool { выделено++; Буфер(it) }
        repeat(1_000_000) { кадр -> пул.взять((кадр % 5) * 20) }
        assertEquals(5, выделено)
        assertEquals(5, пул.создано)
    }

    @Test
    fun `нулевой размер тоже кэшируется`() {
        var выделено = 0
        val пул = BufferPool { выделено++; Буфер(it) }
        repeat(10) { пул.взять(0) }
        assertEquals(1, выделено)
    }
}
