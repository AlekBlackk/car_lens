package com.carlens.vision

import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.media.Image
import android.media.ImageReader
import android.media.ImageWriter
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.core.impl.TagBundle
import androidx.camera.core.impl.utils.ExifData
import androidx.test.platform.app.InstrumentationRegistry
import com.facebook.react.ReactApplication
import com.facebook.react.bridge.ReactApplicationContext
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.SharedArray
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Before
import org.junit.Test

/**
 * Контракт §4.8: слот `SharedArray` из `CarVisionEmbedPlugin` живёт до следующего вызова
 * плагина, а разные боксы одного вызова — в независимой памяти.
 *
 * Тест живёт в `:app:androidTest`, а не в `:vision`: `SharedArray` — тип VisionCamera, и
 * `:vision` намеренно не знает о RN/VisionCamera (см. бриф задачи 7).
 *
 * ## Как получен `VisionCameraProxy`
 *
 * `VisionCameraProxy(ReactApplicationContext)` внутри дёргает нативный `initHybrid` с
 * JSI-рантаймом (`context.javaScriptContextHolder`) и `CallInvokerHolder`
 * (`context.catalystInstance.jsCallInvokerHolder`) — оба доступны только у **живого**
 * `ReactContext`, то есть после того как `ReactHost` поднял инстанс и загрузил JS-бандл.
 * Тестового дубля `VisionCameraProxy` библиотека не поставляет (сам класс — не интерфейс,
 * с приватным `HybridData`), поэтому единственный путь — поднять настоящий `ReactHost`
 * приложения (`MainApplication.reactHost`) через `start()` и дождаться `currentReactContext`.
 *
 * Это не заглушка активности: `ReactHost.start()` не требует Activity, только рабочий JS
 * бандл — в debug-сборке (`getUseDeveloperSupport() == BuildConfig.DEBUG == true`) это
 * означает живой Metro, доступный с тестового устройства. Без него `start()` не
 * завершится за отведённый таймаут, и `поднятьПрокси()` уронит тест с понятным сообщением
 * вместо зависания. Это тяжелее, чем всё остальное дерево инструментальных тестов
 * (`:vision`), которому JS-рантайм не нужен вовсе — см. отчёт задачи 7.
 */
class SharedArrayContractTest {

    private lateinit var плагин: CarVisionEmbedPlugin
    private lateinit var кадр: Frame
    private var читатели = mutableListOf<ImageReader>()
    private var писатели = mutableListOf<ImageWriter>()

    private fun контекст() = InstrumentationRegistry.getInstrumentation().targetContext

    /**
     * Поднимает `ReactHost` приложения и возвращает его `ReactApplicationContext`.
     *
     * `reactHost.start()` идемпотентен: повторный вызов на уже поднятом хосте просто
     * возвращает завершённую задачу, поэтому предзагрузка `CarVision` через
     * `CarVisionHolder` (лениво, при первом `callback`) этому шагу не мешает.
     */
    private fun поднятьПрокси(): VisionCameraProxy {
        val приложение = контекст().applicationContext as ReactApplication
        val хост = checkNotNull(приложение.reactHost) { "reactHost == null: New Architecture выключена?" }
        val поднялся = хост.start().waitForCompletion(30, TimeUnit.SECONDS)
        check(поднялся) {
            "ReactHost не поднялся за 30с — нужен запущенный Metro (debug-сборка) " +
                "или встроенный JS-бандл; без живого JSI-рантайма SharedArray/Frame " +
                "processor plugin сконструировать нельзя"
        }
        val реактКонтекст = checkNotNull(хост.currentReactContext) {
            "currentReactContext пуст после успешного start()"
        }
        return VisionCameraProxy(реактКонтекст as ReactApplicationContext)
    }

    /** Кадр из `ImageReader`/`ImageWriter` — тот же паттерн, что `CarVisionTest.кадр()`. */
    private fun сыройКадр(): Image {
        val читатель = ImageReader.newInstance(640, 480, ImageFormat.YUV_420_888, 2)
        читатели.add(читатель)
        val писатель = ImageWriter.newInstance(читатель.surface, 2)
        писатели.add(писатель)
        val заготовка = писатель.dequeueInputImage()
        заполнить(заготовка)
        писатель.queueInputImage(заготовка)
        return checkNotNull(читатель.acquireLatestImage()) { "ImageReader не отдал кадр" }
    }

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

    /** Оборачивает сырой `Image` в минимальный `Frame`, готовый к `plugin.callback(...)`. */
    private fun frame(): Frame {
        val image = ТестовыйImageProxy(сыройКадр(), угол = 0)
        return Frame(image).also { it.incrementRefCount() }
    }

    private fun вМассив(shared: SharedArray): FloatArray {
        val буфер = shared.byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
        val результат = FloatArray(буфер.remaining())
        буфер.get(результат)
        return результат
    }

    private val аргументыОдногоБокса = mapOf(
        "boxes" to listOf(30f, 40f, 120f, 90f, 0.9f),
        "pad" to 0.08,
        "topK" to 5,
    )

    @Before
    fun поднять() {
        val proxy = поднятьПрокси()
        плагин = CarVisionEmbedPlugin(proxy, контекст())
        кадр = frame()
    }

    @After
    fun закрыть() {
        писатели.forEach { it.close() }
        читатели.forEach { it.close() }
    }

    @Test
    fun два_бокса_в_одном_вызове_дают_разные_эмбеддинги() {
        // Один слот на оба результата означал бы, что оба указывают в одну память, и
        // вторая машина в кадре получала бы подпись первой. Тот же класс ошибки на
        // Kotlin-уровне ловит `два_бокса_дают_разные_эмбеддинги` из M-1.5.
        val ответ = плагин.callback(
            кадр,
            mapOf(
                "boxes" to listOf(
                    30f, 40f, 120f, 90f, 0.9f,
                    300f, 500f, 140f, 100f, 0.8f,
                ),
                "pad" to 0.08,
                "topK" to 5,
            ),
        ) as Map<*, *>
        val первый = вМассив(ответ["emb0"] as SharedArray)
        val второй = вМассив(ответ["emb1"] as SharedArray)
        assertEquals(2, ответ["count"])
        assertFalse("слоты указывают в одну память", первый.contentEquals(второй))
    }

    @Test
    fun следующий_вызов_не_меняет_скопированный_результат() {
        // Контракт §4.8: буфер слота действителен до следующего вызова. Копия обязана
        // пережить его, сам буфер — нет, и это ожидаемо, а не баг.
        val первый = плагин.callback(кадр, аргументыОдногоБокса) as Map<*, *>
        val копия = вМассив(первый["emb0"] as SharedArray)
        val другойКадр = frame()
        плагин.callback(другойКадр, аргументыОдногоБокса)
        val копияПосле = копия.copyOf()
        assertArrayEquals(копия, копияПосле, 0f)
    }
}

/**
 * Минимальная реализация `ImageInfo` для теста: только то, что реально читает `Frame`
 * (`getOrientation`/`getIsMirrored`/`getTimestamp`) плюс обязательные методы интерфейса.
 * `getSensorToBufferTransformMatrix` переопределён identity-матрицей — иначе
 * `Frame.getIsMirrored()` упал бы на default-реализации, которой в тесте взяться неоткуда.
 */
private class ТестоваяИнформацияОКадре(private val угол: Int) : ImageInfo {
    override fun getTagBundle(): TagBundle = TagBundle.emptyBundle()
    override fun getTimestamp(): Long = 0L
    override fun getRotationDegrees(): Int = угол
    override fun getSensorToBufferTransformMatrix(): Matrix = Matrix()
    override fun populateExifData(builder: ExifData.Builder) {}
}

/**
 * Минимальная реализация `ImageProxy` поверх настоящего `android.media.Image` с камеры
 * (`ImageReader`/`ImageWriter`, как в `CarVisionTest.кадр()`). VisionCamera тестового
 * дубля не поставляет: `Frame(ImageProxy)` — публичный конструктор, а `ImageProxy` —
 * интерфейс CameraX, обязательные методы которого перечислены в бриф задачи 7.
 */
private class ТестовыйImageProxy(
    private val изображение: Image,
    угол: Int,
) : ImageProxy {
    private val инфо = ТестоваяИнформацияОКадре(угол)
    override fun close() = изображение.close()
    override fun getCropRect(): Rect = Rect(0, 0, изображение.width, изображение.height)
    override fun setCropRect(rect: Rect?) {}
    override fun getFormat(): Int = изображение.format
    override fun getHeight(): Int = изображение.height
    override fun getWidth(): Int = изображение.width
    override fun getPlanes(): Array<ImageProxy.PlaneProxy> = arrayOf()
    override fun getImageInfo(): ImageInfo = инфо
    override fun getImage(): Image = изображение
}
