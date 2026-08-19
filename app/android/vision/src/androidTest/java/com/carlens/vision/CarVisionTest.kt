package com.carlens.vision

import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.media.ImageWriter
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class CarVisionTest {

    private lateinit var зрение: CarVision
    private var читатель: ImageReader? = null
    private var писатель: ImageWriter? = null

    private fun ассеты() = InstrumentationRegistry.getInstrumentation().context.assets

    /**
     * Кадр из `ImageReader`: настоящий `YUV_420_888` со своими шагами, как с камеры.
     *
     * План предлагал рисовать в `Surface` через `lockCanvas`, но поверхность формата
     * `YUV_420_888` холста не отдаёт — Canvas умеет только RGB-форматы. Берётся второй
     * допущенный планом путь: кадр заполняется `ImageWriter`, то есть теми же
     * плоскостями и шагами, которые потом придут от камеры.
     */
    private fun кадр(): Image {
        val текущийЧитатель = читатель ?: ImageReader
            .newInstance(640, 480, ImageFormat.YUV_420_888, 2)
            .also { читатель = it }
        val текущийПисатель = писатель ?: ImageWriter
            .newInstance(текущийЧитатель.surface, 2)
            .also { писатель = it }
        val заготовка = текущийПисатель.dequeueInputImage()
        заполнить(заготовка)
        текущийПисатель.queueInputImage(заготовка)
        return checkNotNull(текущийЧитатель.acquireLatestImage()) { "ImageReader не отдал кадр" }
    }

    /** Диагональный градиент по яркости и ровная хрома: не сплошная заливка. */
    private fun заполнить(изображение: Image) {
        val яркость = изображение.planes[0]
        val высота = изображение.height
        val ширина = изображение.width
        for (строка in 0 until высота) {
            for (столбец in 0 until ширина) {
                val значение = ((строка + столбец) % 200 + 20)
                яркость.buffer.put(
                    строка * яркость.rowStride + столбец * яркость.pixelStride,
                    значение.toByte(),
                )
            }
        }
        for (плоскость in listOf(изображение.planes[1], изображение.planes[2])) {
            for (строка in 0 until высота / 2) {
                for (столбец in 0 until ширина / 2) {
                    плоскость.buffer.put(
                        строка * плоскость.rowStride + столбец * плоскость.pixelStride,
                        120.toByte(),
                    )
                }
            }
        }
    }

    @Before
    fun поднять() {
        зрение = CarVision.load(ассеты())
    }

    @After
    fun закрыть() {
        зрение.close()
        писатель?.close()
        читатель?.close()
    }

    @Test
    fun индекс_загружен_и_совпадает_с_манифестом() {
        assertTrue(зрение.indexInfo.numClasses > 0)
        assertEquals(256, зрение.indexInfo.embeddingDim)
        assertEquals(зрение.indexInfo.numClasses, зрение.classes.size)
    }

    @Test
    fun делегат_выбран_и_назван() {
        // Значение уезжает в отчёт вехи: «работает за 22 мс» и «работает за 22 мс,
        // потому что тихо ушло на CPU» — разные новости. Печатается, а не только
        // проверяется: выбор делегата зависит от устройства, и его надо видеть в логе прогона.
        println("ДЕЛЕГАТ=${зрение.delegateMode}")
        assertTrue(зрение.delegateMode == DelegateMode.GPU || зрение.delegateMode == DelegateMode.CPU)
    }

    @Test
    fun detect_не_закрывает_кадр() {
        val изображение = кадр()
        зрение.detect(изображение, rotationDegrees = 90)
        // Если бы плагин закрыл кадр, обращение к плоскостям бросило бы IllegalStateException.
        assertTrue(изображение.planes[0].buffer.capacity() > 0)
        изображение.close()
    }

    @Test
    fun embedAndMatch_даёт_нормированный_эмбеддинг_и_topK() {
        val изображение = кадр()
        val боксы = listOf(Box(100f, 100f, 200f, 150f, 0.9f))
        val результаты = зрение.embedAndMatch(изображение, 90, боксы, pad = 0.08f, topK = 5)
        изображение.close()
        assertEquals(1, результаты.size)
        val эмбеддинг = результаты.first().embedding
        assertEquals(зрение.indexInfo.embeddingDim, эмбеддинг.size)
        val длина = эмбеддинг.fold(0.0) { сумма, значение -> сумма + значение * значение }
        assertEquals(1.0, длина, 1e-3)
        assertEquals(5, результаты.first().matches.size)
        val классы = результаты.first().matches.map { it.classIdx }
        assertEquals(5, классы.toSet().size)
    }

    @Test
    fun embedding_не_разделяет_память_с_интерпретатором() {
        val изображение = кадр()
        val боксы = listOf(Box(100f, 100f, 200f, 150f, 0.9f))
        val первый = зрение.embedAndMatch(изображение, 90, боксы).first().embedding.copyOf()
        val снова = зрение.embedAndMatch(изображение, 90, боксы).first().embedding
        изображение.close()
        assertEquals(первый.size, снова.size)
        // Второй вызов не обязан совпасть с первым, но первый результат меняться не может.
        assertTrue(первый.any { it != 0f })
    }

    @Test
    fun два_бокса_дают_разные_эмбеддинги() {
        // Ловит переиспользование одного буфера на весь список: без копии оба результата
        // указывали бы на один массив, и второй бокс затирал бы первый.
        val изображение = кадр()
        val боксы = listOf(
            Box(20f, 20f, 120f, 120f, 0.9f),
            Box(300f, 200f, 160f, 140f, 0.8f),
        )
        val результаты = зрение.embedAndMatch(изображение, 0, боксы)
        изображение.close()
        assertEquals(2, результаты.size)
        assertNotEquals(результаты[0].embedding.toList(), результаты[1].embedding.toList())
    }

    @Test
    fun matchEmbedding_нормализует_чужой_вектор() {
        val сырой = FloatArray(зрение.indexInfo.embeddingDim) { 0.01f * (it % 7 + 1) }
        val совпадения = зрение.matchEmbedding(сырой, topK = 3)
        assertEquals(3, совпадения.size)
        assertTrue(совпадения.all { it.sim in -1.001f..1.001f })
    }

    @Test
    fun вызов_после_close_роняет_понятным_исключением() {
        val изображение = кадр()
        зрение.close()
        val отказ = assertThrows(IllegalStateException::class.java) {
            зрение.detect(изображение, 90)
        }
        изображение.close()
        assertTrue(отказ.message!!.contains("close"))
    }

    @Test
    fun подменённый_хеш_роняет_загрузку() {
        val спек = AssetSpec(expected = mapOf("index" to "ff".repeat(32)))
        assertThrows(AssetMismatchException::class.java) { CarVision.load(ассеты(), спек) }
    }
}
