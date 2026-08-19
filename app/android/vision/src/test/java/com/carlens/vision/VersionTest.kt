package com.carlens.vision

import org.junit.Assert.assertEquals
import org.junit.Test

class VersionTest {
    @Test
    fun `версия формата индекса совпадает с 13_2 спеки`() {
        assertEquals(1, INDEX_FORMAT_VERSION)
    }
}
