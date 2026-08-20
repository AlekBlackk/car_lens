package com.carlens.vision

import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.media.ImageWriter
import android.os.Build
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.ceil

/**
 * Замер латентности по стадиям — отчёт вехи, а не порог сборки.
 *
 * Числа печатаются в logcat и переносятся в `bench/latency.md` руками: они зависят от
 * телефона, и держать их в ассертах значило бы красить сборку в цвет чужого железа.
 * Единственный ассерт — kNN: он про арифметику на Kotlin, а не про графы, и именно им
 * закрывается открытый вопрос решения Р-2. Порог §5.4 на трёх устройствах проверяется
 * на M5.
 *
 * Прогонов два, по делегату на каждый: делегат выбирается по устройству, и без
 * принудительного CPU вторая половина таблицы не снимается вовсе.
 */
class LatencyTest {

    private var зрение: CarVision? = null
    private var читатель: ImageReader? = null
    private var писатель: ImageWriter? = null

    private fun ассеты() = InstrumentationRegistry.getInstrumentation().context.assets

    @After
    fun закрыть() {
        зрение?.close()
        писатель?.close()
        читатель?.close()
    }

    @Test
    fun латентность_по_стадиям_на_делегате_по_умолчанию() {
        замерить(AssetSpec())
    }

    @Test
    fun латентность_по_стадиям_на_принудительном_CPU() {
        замерить(AssetSpec(allowGpu = false))
    }

    private fun замерить(спек: AssetSpec) {
        val зрение = CarVision.load(ассеты(), спек).also { this.зрение = it }
        // Препроцессинг меряется отдельно от графа тем же кодом, что зовёт `detect`:
        // стадия «детектор» — это выборка из плоскостей плюс интерпретатор, и без
        // разделения непонятно, куда смотреть, если стадия не влезает в бюджет.
        val манифест = ModelsManifest.parse(
            ассеты().open(спек.manifestName).use { it.readBytes() }
        )
        val спекТензора = манифест.detectorSpec()
        val буферПикселей = FloatArray(спекТензора.floatCount)
        val кадры = 200
        val прогрев = 20
        val препроцессинг = ArrayList<Float>(кадры)
        val детектор = ArrayList<Float>(кадры)
        val классификатор = ArrayList<Float>(кадры)
        val knn = ArrayList<Float>(кадры)
        val итог = ArrayList<Float>(кадры)
        repeat(кадры + прогрев) { номер ->
            val изображение = кадр()
            val tп = SystemClock.elapsedRealtimeNanos()
            FrameConvert.letterbox(
                вКадр(изображение), спекТензора, манифест.detector.letterboxPadValue, буферПикселей,
            )
            val t0 = SystemClock.elapsedRealtimeNanos()
            val боксы = зрение.detect(изображение, 90)
            val t1 = SystemClock.elapsedRealtimeNanos()
            // На синтетическом кадре машин нет, и без запасного бокса стадии
            // классификатора и kNN мерили бы пустой список, то есть ничего.
            val один = боксы.take(1).ifEmpty { listOf(ЗАПАСНОЙ_БОКС) }
            val результаты = зрение.embedAndMatch(изображение, 90, один)
            val t2 = SystemClock.elapsedRealtimeNanos()
            результаты.firstOrNull()?.let { зрение.matchEmbedding(it.embedding, topK = 5) }
            val t3 = SystemClock.elapsedRealtimeNanos()
            изображение.close()
            if (номер >= прогрев) {
                препроцессинг.add((t0 - tп) / 1_000_000f)
                детектор.add((t1 - t0) / 1_000_000f)
                классификатор.add((t2 - t1) / 1_000_000f)
                knn.add((t3 - t2) / 1_000_000f)
                итог.add((t3 - t0) / 1_000_000f)
            }
        }
        println("УСТРОЙСТВО=${Build.MANUFACTURER} ${Build.MODEL} android=${Build.VERSION.RELEASE}")
        println("ДЕЛЕГАТ=${зрение.delegateMode} классов=${зрение.indexInfo.numClasses}")
        печать("препроцессинг", препроцессинг)
        печать("детектор", детектор)
        печать("классификатор", классификатор)
        печать("kNN", knn)
        печать("итог", итог)
        assertTrue(
            "kNN вышел за бюджет §5.4: p95=${перцентиль(knn, 95)} мс",
            перцентиль(knn, 95) < 5f,
        )
    }

    private fun печать(стадия: String, замеры: List<Float>) {
        println("ЗАМЕР $стадия p50=${перцентиль(замеры, 50)} p95=${перцентиль(замеры, 95)}")
    }

    /**
     * Перцентиль ближайшим рангом, без интерполяции.
     *
     * Интерполяция между соседними замерами усредняет два разных кадра в число, которого
     * не было ни на одном: для хвоста времени кадра это ровно та часть, ради которой
     * хвост и меряют.
     */
    private fun перцентиль(замеры: List<Float>, процент: Int): Float {
        require(замеры.isNotEmpty()) { "перцентиль пустой выборки" }
        val отсортированные = замеры.sorted()
        val ранг = ceil(процент / 100.0 * отсортированные.size).toInt().coerceAtLeast(1)
        return отсортированные[ранг - 1]
    }

    /**
     * Кадр `YUV_420_888` с настоящими шагами плоскостей, как с камеры.
     *
     * Тот же путь, что в `CarVisionTest`: поверхность этого формата холста не отдаёт,
     * поэтому кадр заполняется `ImageWriter`.
     */
    private fun кадр(): Image {
        val текущийЧитатель = читатель ?: ImageReader
            .newInstance(1280, 720, ImageFormat.YUV_420_888, 2)
            .also { читатель = it }
        val текущийПисатель = писатель ?: ImageWriter
            .newInstance(текущийЧитатель.surface, 2)
            .also { писатель = it }
        val заготовка = текущийПисатель.dequeueInputImage()
        заполнить(заготовка)
        текущийПисатель.queueInputImage(заготовка)
        return checkNotNull(текущийЧитатель.acquireLatestImage()) { "ImageReader не отдал кадр" }
    }

    /** Те же плоскости без копирования, что строит фасад: замер обязан мерить его путь. */
    private fun вКадр(изображение: Image): YuvFrame {
        val яркость = изображение.planes[0]
        val синева = изображение.planes[1]
        val краснота = изображение.planes[2]
        return YuvFrame(
            width = изображение.width,
            height = изображение.height,
            rotationDegrees = 90,
            y = яркость.buffer,
            yRowStride = яркость.rowStride,
            yPixelStride = яркость.pixelStride,
            u = синева.buffer,
            v = краснота.buffer,
            uvRowStride = синева.rowStride,
            uvPixelStride = синева.pixelStride,
        )
    }

    /** Диагональный градиент по яркости и ровная хрома: не сплошная заливка. */
    private fun заполнить(изображение: Image) {
        val яркость = изображение.planes[0]
        for (строка in 0 until изображение.height) {
            for (столбец in 0 until изображение.width) {
                яркость.buffer.put(
                    строка * яркость.rowStride + столбец * яркость.pixelStride,
                    ((строка + столбец) % 200 + 20).toByte(),
                )
            }
        }
        for (плоскость in listOf(изображение.planes[1], изображение.planes[2])) {
            for (строка in 0 until изображение.height / 2) {
                for (столбец in 0 until изображение.width / 2) {
                    плоскость.buffer.put(
                        строка * плоскость.rowStride + столбец * плоскость.pixelStride,
                        120.toByte(),
                    )
                }
            }
        }
    }

    private companion object {
        /**
         * Половина выпрямленного кадра по центру: правдоподобный кроп, а не вырожденный.
         *
         * Кадр снимается 1280x720 и подаётся с `rotationDegrees = 90`, то есть
         * выпрямленный размер — 720x1280, и координаты здесь в нём.
         */
        val ЗАПАСНОЙ_БОКС = Box(180f, 320f, 360f, 640f, 0.9f)
    }
}
