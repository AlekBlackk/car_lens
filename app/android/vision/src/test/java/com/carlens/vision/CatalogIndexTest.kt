package com.carlens.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.sqrt

class CatalogIndexTest {

    private fun ресурс(имя: String): ByteArray =
        checkNotNull(javaClass.classLoader!!.getResourceAsStream(имя)) { "нет ресурса $имя" }
            .use { it.readBytes() }

    private fun бинарник() = ресурс("catalog_index.bin")
    private fun мета() = ресурс("catalog_meta.json")

    private fun разобрать(
        bin: ByteArray = бинарник(),
        meta: ByteArray = мета(),
    ): CatalogIndex = CatalogIndex.parse(bin, meta, Embeddings::sha256)

    @Test
    fun `заголовок фикстуры разобран по 13_2`() {
        val индекс = разобрать()
        assertEquals(1, индекс.header.formatVersion)
        assertEquals(1, индекс.header.dtype)
        assertTrue(индекс.header.normalized)
        assertEquals(индекс.header.numClasses, индекс.classes.size)
    }

    @Test
    fun `поля класса разобраны под именами формата`() {
        val классы = разобрать().classes
        assertEquals("acme__one", классы[0].classId)
        assertEquals(listOf("acme__one"), классы[0].members)
        assertEquals("Acme", классы[0].brand)
        assertEquals("Acme One", классы[0].display)
    }

    @Test
    fun `у разнобрендовой группы марки нет, а участники на месте`() {
        val группа = разобрать().classes[1]
        assertEquals("grp__twins", группа.classId)
        assertEquals(listOf("acme__two", "beta__two"), группа.members)
        assertNull(группа.brand)
        assertEquals("Acme Two / Beta Two", группа.display)
    }

    @Test
    fun `мета старого формата отвергается ошибкой формата индекса`() {
        // Класс равен модели: `model_id` вместо `class_id`, `members` нет вовсе.
        val старая = (
            "{\"classes\":[{\"model_id\":\"acme__one\",\"brand\":\"Acme\"," +
                "\"display\":\"Acme One\",\"platform_group\":null}]}"
            ).toByteArray(Charsets.UTF_8)
        val отказ = assertThrows(IndexFormatException::class.java) {
            разобрать(bin = бинарникПодМету(старая, классов = 1), meta = старая)
        }
        // Сообщение обязано называть и файл, и пропавшее поле: иначе на устройстве видно
        // «что-то сломалось в сериализации», а не «мета другого формата».
        assertTrue(отказ.message!!.contains("catalog_meta.json"))
        assertTrue(отказ.message!!.contains("class_id"))
    }

    @Test
    fun `центроид читается по формуле смещения`() {
        val индекс = разобрать()
        val центроид = индекс.centroid(0, 0)
        assertEquals(индекс.header.embeddingDim, центроид.size)
        val длина = центроид.sumOf { (it * it).toDouble() }
        assertEquals(1.0, длина, 1e-4)
    }

    @Test
    fun `чужой magic роняет`() {
        val испорченный = бинарник().also { it[0] = 'X'.code.toByte() }
        val отказ = assertThrows(IndexFormatException::class.java) { разобрать(bin = испорченный) }
        assertTrue(отказ.message!!.contains("CLIX"))
    }

    @Test
    fun `чужая версия формата роняет`() {
        val испорченный = бинарник()
        ByteBuffer.wrap(испорченный).order(ByteOrder.LITTLE_ENDIAN).putShort(4, 2)
        assertThrows(IndexFormatException::class.java) { разобрать(bin = испорченный) }
    }

    @Test
    fun `float16 роняет отдельным сообщением`() {
        val испорченный = бинарник().also { it[16] = 0 }
        val отказ = assertThrows(IndexFormatException::class.java) { разобрать(bin = испорченный) }
        assertTrue(отказ.message!!.contains("float16"))
    }

    @Test
    fun `ненулевой reserved роняет`() {
        val испорченный = бинарник().also { it[53] = 7 }
        assertThrows(IndexFormatException::class.java) { разобрать(bin = испорченный) }
    }

    @Test
    fun `длина против заголовка роняет`() {
        val короткий = бинарник().copyOf(бинарник().size - 4)
        val отказ = assertThrows(IndexFormatException::class.java) { разобрать(bin = короткий) }
        assertTrue(отказ.message!!.contains(короткий.size.toString()))
    }

    @Test
    fun `чужая мета роняет по sha256`() {
        val чужая = мета().let { String(it, Charsets.UTF_8) + " " }.toByteArray(Charsets.UTF_8)
        val отказ = assertThrows(IndexFormatException::class.java) { разобрать(meta = чужая) }
        assertTrue(отказ.message!!.contains("sha256"))
    }

    @Test
    fun `topK больше числа классов усекается`() {
        val индекс = разобрать()
        val совпадения = индекс.knn(индекс.centroid(0, 0), topK = индекс.header.numClasses + 10)
        assertEquals(индекс.header.numClasses, совпадения.size)
    }

    @Test
    fun `чужая размерность запроса роняет`() {
        val индекс = разобрать()
        assertThrows(IndexFormatException::class.java) { индекс.knn(FloatArray(7), topK = 5) }
    }

    // Тесты kNN идут на синтетическом индексе, а не на фикстуре: фикстура даёт два
    // класса, а ловушка «top-K по парам класс×центроид» ловится только когда лучших
    // центроидов одного класса больше, чем классов в ответе. Фикстура при этом
    // остаётся эталоном формата — её параметры дословно закреплены в
    // `app/src/vision/catalogIndex.test.ts` и сверяются `tests/test_index_fixture.py`.

    @Test
    fun `top-K это разные классы, а не центроиды одного`() {
        val индекс = синтетический()
        val совпадения = индекс.knn(запросПоПервойОси(), topK = 5)
        // Класс 0 держит три лучших центроида всего индекса (1.0, 0.99, 0.98): отбор по
        // парам вместо максимума по классу вернул бы его три раза подряд.
        assertEquals(listOf(0, 2, 4, 5, 1), совпадения.map { it.classIdx })
        assertEquals(5, совпадения.map { it.classIdx }.toSet().size)
        assertEquals(1.0f, совпадения[0].sim, 1e-5f)
        assertEquals(0.95f, совпадения[1].sim, 1e-5f)
        assertEquals(0.6f, совпадения[4].sim, 1e-5f)
    }

    @Test
    fun `сходства идут по невозрастанию`() {
        val индекс = синтетический()
        val совпадения = индекс.knn(индекс.centroid(3, 1), topK = 5)
        val подряд = совпадения.zipWithNext().all { (первое, второе) -> первое.sim >= второе.sim }
        assertTrue(подряд)
    }

    @Test
    fun `topK меньше единицы роняет`() {
        val индекс = синтетический()
        val отказ = assertThrows(IndexFormatException::class.java) {
            индекс.knn(запросПоПервойОси(), topK = 0)
        }
        assertTrue(отказ.message!!.contains("0"))
        assertThrows(IndexFormatException::class.java) {
            индекс.knn(запросПоПервойОси(), topK = -1)
        }
    }

    // Класс без эталонных кадров записан в индексе нулями (§13.2, `stack_centroids`
    // с `allow_empty=True`): косинус к нулевому центроиду равен ровно 0, и этот ноль
    // бьёт отрицательное сходство живого класса. Семантика та же, что в питоновской
    // читалке (`pipeline/knn.py`, `ranked_by_centroids`).

    @Test
    fun `класс без эталона не вытесняет живой`() {
        val совпадения = сМёртвымКлассом().knn(запросПротивВсехОсей(), topK = 2)
        assertEquals(listOf(0, 1), совпадения.map { it.classIdx })
        assertEquals(-0.5f, совпадения[0].sim, 1e-5f)
        assertEquals(-0.5f, совпадения[1].sim, 1e-5f)
    }

    @Test
    fun `мёртвый класс уходит в хвост, а не исчезает`() {
        // Контракт «topK разных классов» сильнее: слот заполняется, но последним.
        val совпадения = сМёртвымКлассом().knn(запросПротивВсехОсей(), topK = 3)
        assertEquals(3, совпадения.map { it.classIdx }.toSet().size)
        assertEquals(2, совпадения[2].classIdx)
        assertEquals(Float.NEGATIVE_INFINITY, совпадения[2].sim, 0f)
    }

    @Test
    fun `индекс без живых классов отвергается`() {
        val мёртвый = собрать(Array(3) { Array(2) { нулевой() } })
        val отказ = assertThrows(IndexFormatException::class.java) {
            мёртвый.knn(запросПротивВсехОсей(), topK = 2)
        }
        assertTrue(отказ.message!!.contains("ни одного класса с эталоном"))
    }

    // Косинусы синтетических центроидов с `запросПоПервойОси`. Максимумы по классам —
    // 1.0, 0.6, 0.95, 0.5, 0.9, 0.7: порядок классов в ответе не совпадает с порядком
    // их номеров, иначе тест прошёл бы и на реализации, которая не сортирует.
    private val косинусы = arrayOf(
        floatArrayOf(1.0f, 0.99f, 0.98f),
        floatArrayOf(0.6f, 0.55f, 0.5f),
        floatArrayOf(0.95f, 0.4f, 0.3f),
        floatArrayOf(0.5f, 0.45f, 0.2f),
        floatArrayOf(0.9f, 0.85f, 0.1f),
        floatArrayOf(0.7f, 0.65f, 0.05f),
    )

    private fun запросПоПервойОси() = FloatArray(СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ).also { it[0] = 1f }

    /** Запрос против всех осей: у живых классов сходство отрицательное, у мёртвого — ноль. */
    private fun запросПротивВсехОсей() = FloatArray(СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ) { -0.5f }

    private fun ось(номер: Int) =
        FloatArray(СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ).also { it[номер] = 1f }

    private fun нулевой() = FloatArray(СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ)

    /** Два живых класса по осям и один без эталонных кадров — нулями, как пишет §13.2. */
    private fun сМёртвымКлассом() = собрать(
        arrayOf(
            arrayOf(ось(0), ось(1)),
            arrayOf(ось(2), ось(3)),
            arrayOf(нулевой(), нулевой()),
        )
    )

    /** Единичный вектор с заданным косинусом к `запросПоПервойОси`. */
    private fun вектор(косинус: Float): FloatArray {
        val результат = FloatArray(СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ)
        результат[0] = косинус
        результат[1] = sqrt(1f - косинус * косинус)
        return результат
    }

    /**
     * Собрать индекс §13.2 в памяти из готовых центроидов (класс×центроид×вектор).
     * Байты пишутся руками, а не через `CatalogIndex`: билдер, зовущий разбор, проверял бы
     * сам себя, а не парсер.
     */
    private fun собрать(центроиды: Array<Array<FloatArray>>): CatalogIndex {
        val классов = центроиды.size
        val центроидов = центроиды[0].size
        val мета = (0 until классов).joinToString(
            separator = ",",
            prefix = "{\"classes\":[",
            postfix = "]}",
        ) { номер ->
            "{\"class_id\":\"synth__$номер\",\"members\":[\"synth__$номер\"]," +
                "\"brand\":\"Synth\",\"display\":\"Synth $номер\"}"
        }.toByteArray(Charsets.UTF_8)

        val значений = классов * центроидов * СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ
        val буфер = ByteBuffer.allocate(64 + значений * 4).order(ByteOrder.LITTLE_ENDIAN)
        записатьЗаголовок(буфер, мета, классов, центроидов)
        буфер.position(64)
        for (класс in центроиды) {
            for (центроид in класс) центроид.forEach { буфер.putFloat(it) }
        }
        return CatalogIndex.parse(буфер.array(), мета, Embeddings::sha256)
    }

    private fun синтетический(): CatalogIndex = собрать(
        Array(косинусы.size) { класс ->
            Array(косинусы[класс].size) { центроид -> вектор(косинусы[класс][центроид]) }
        }
    )

    /**
     * Бинарник, валидный по заголовку и sha256 под заданную мету: разбор доходит до разбора
     * JSON и падает уже на нём, а не раньше. Тело нулевое — до него дело не доходит.
     */
    private fun бинарникПодМету(мета: ByteArray, классов: Int): ByteArray {
        val центроидов = 1
        val значений = классов * центроидов * СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ
        val буфер = ByteBuffer.allocate(64 + значений * 4).order(ByteOrder.LITTLE_ENDIAN)
        записатьЗаголовок(буфер, мета, классов, центроидов)
        return буфер.array()
    }

    private fun записатьЗаголовок(
        буфер: ByteBuffer,
        мета: ByteArray,
        классов: Int,
        центроидов: Int,
    ) {
        "CLIX".toByteArray(Charsets.US_ASCII).forEachIndexed { сдвиг, байт -> буфер.put(сдвиг, байт) }
        буфер.putShort(4, INDEX_FORMAT_VERSION.toShort())
        буфер.putShort(6, СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ.toShort())
        буфер.putInt(8, классов)
        буфер.putInt(12, центроидов)
        буфер.put(16, 1) // dtype: float32
        буфер.put(17, 1) // normalized
        MessageDigest.getInstance("SHA-256").digest(мета).forEachIndexed { сдвиг, байт ->
            буфер.put(20 + сдвиг, байт)
        }
    }

    private companion object {
        const val СИНТЕТИЧЕСКАЯ_РАЗМЕРНОСТЬ = 4
    }
}
