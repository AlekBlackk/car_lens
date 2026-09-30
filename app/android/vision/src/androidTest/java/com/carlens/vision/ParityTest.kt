package com.carlens.vision

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Паритет с питоновским конвейером в двух уровнях (§9 дизайна).
 *
 * Диагностика при провале: падает A — виноват рантайм, нормализация или L2; A зелёный,
 * B красный — виноват кроп или ресайз. Разбираться в обратном порядке бессмысленно.
 */
class ParityTest {

    private lateinit var зрение: CarVision
    private lateinit var эталон: JSONObject

    private fun ассеты() = InstrumentationRegistry.getInstrumentation().context.assets

    @Before
    fun поднять() {
        // Косинус ≥ 0.9999 к питону держит только CPU: активации GPU-делегата fp16. Согласие GPU
        // с CPU на решениях меряет `GpuAgreementTest`, а не этот тест.
        зрение = CarVision.load(ассеты(), AssetSpec(classifierGpu = false))
        эталон = JSONObject(
            String(ассеты().open("golden.json").use { it.readBytes() }, Charsets.UTF_8)
        )
    }

    @After
    fun закрыть() = зрение.close()

    private fun косинус(первый: FloatArray, второй: FloatArray): Float {
        var сумма = 0f
        for (индекс in первый.indices) сумма += первый[индекс] * второй[индекс]
        return сумма
    }

    private fun картинка(имя: String) = ассеты().open(имя).use { BitmapFactory.decodeStream(it) }

    private fun боксИз(запись: JSONObject, поле: String): Box = запись.getJSONObject(поле).let {
        Box(
            it.getDouble("x").toFloat(), it.getDouble("y").toFloat(),
            it.getDouble("w").toFloat(), it.getDouble("h").toFloat(), 0.9f,
        )
    }

    /** Пиксели эталонного кропа как NHWC RGB 0..255 — в той же раскладке, что отдаёт плагин. */
    private fun пикселиИзPNG(имя: String): FloatArray {
        val картинка = картинка(имя)
        val ширина = картинка.width
        val высота = картинка.height
        val сырые = IntArray(ширина * высота)
        картинка.getPixels(сырые, 0, ширина, 0, 0, ширина, высота)
        val результат = FloatArray(ширина * высота * 3)
        for (позиция in сырые.indices) {
            val цвет = сырые[позиция]
            результат[позиция * 3] = ((цвет shr 16) and 0xFF).toFloat()
            результат[позиция * 3 + 1] = ((цвет shr 8) and 0xFF).toFloat()
            результат[позиция * 3 + 2] = (цвет and 0xFF).toFloat()
        }
        return результат
    }

    private fun ожидаемыйВектор(запись: JSONObject, размер: Int): FloatArray {
        val массив = запись.getJSONArray("embedding")
        return FloatArray(размер) { массив.getDouble(it).toFloat() }
    }

    @Test
    fun эталон_собран_на_тех_же_моделях() {
        // Эталон от другого прогона разошёлся бы по эмбеддингам, и оба уровня показали
        // бы расхождение конвейеров там, где на деле разошлись веса.
        val манифест = JSONObject(
            String(
                ассеты().open("models_manifest.json").use { it.readBytes() },
                Charsets.UTF_8,
            )
        )
        assertEquals(
            манифест.getJSONObject("files").getJSONObject("classifier").getString("sha256"),
            эталон.getString("classifier_sha256"),
        )
        assertEquals(
            манифест.getJSONObject("provenance").getString("checkpoint_sha256"),
            эталон.getString("checkpoint_sha256"),
        )
        assertEquals(зрение.indexInfo.embeddingDim, эталон.getInt("embedding_dim"))
    }

    @Test
    fun уровень_A_тензор_совпадает_с_питоном() {
        val записи = эталон.getJSONArray("tensor")
        for (индекс in 0 until записи.length()) {
            val запись = записи.getJSONObject(индекс)
            val эмбеддинг = зрение.embedBitmapForTest(картинка(запись.getString("png")))
            val сходство = косинус(эмбеддинг, ожидаемыйВектор(запись, эмбеддинг.size))
            assertTrue("кадр $индекс: косинус $сходство < 0.9999", сходство >= 0.9999f)
            assertEquals(
                "кадр $индекс: top-1 разошёлся",
                запись.getJSONArray("top5").getInt(0),
                зрение.matchEmbedding(эмбеддинг, topK = 5).first().classIdx,
            )
        }
    }

    @Test
    fun уровень_B_кроп_и_ресайз_совпадают_с_питоном() {
        // Сравниваются пиксели входа, а не эмбеддинги: классификатор квантованный, и
        // микроразница двух честных реализаций билинейной выборки расходится в косинус
        // 0.97, тогда как настоящая ошибка кропа даёт 0.87-0.99 — по косинусу эти
        // случаи неразличимы. По пикселям разделение чистое: честный путь укладывается
        // в среднюю |Δ| 3.9 уровня яркости, сдвиг области на два пикселя даёт от 5.7.
        val записи = эталон.getJSONArray("frame")
        for (индекс in 0 until записи.length()) {
            val запись = записи.getJSONObject(индекс)
            val бокс = боксИз(запись, "box")
            val свои = зрение.cropPixelsForTest(картинка(запись.getString("png")), бокс, pad = 0.08f)
            val ожидаемые = пикселиИзPNG(запись.getString("tensor_png"))
            assertEquals("кадр $индекс: размер тензора", ожидаемые.size, свои.size)
            var сумма = 0.0
            for (позиция in свои.indices) сумма += kotlin.math.abs(свои[позиция] - ожидаемые[позиция])
            val средняя = сумма / свои.size
            assertTrue("кадр $индекс: средняя разница пикселей $средняя > 5.0", средняя <= 5.0)
        }
    }

    @Test
    fun уровень_B_класс_совпадает_с_питоном() {
        val записи = эталон.getJSONArray("frame")
        for (индекс in 0 until записи.length()) {
            val запись = записи.getJSONObject(индекс)
            val результат = зрение.embedBitmapCropForTest(
                картинка(запись.getString("png")), боксИз(запись, "box"), pad = 0.08f, topK = 5,
            )
            val сходство = косинус(результат.embedding, ожидаемыйВектор(запись, результат.embedding.size))
            // Порог грубый по той же причине: на INT8 честный путь даёт от 0.94.
            assertTrue("кадр $индекс: косинус $сходство < 0.9", сходство >= 0.9f)
            assertEquals(
                "кадр $индекс: top-1 разошёлся",
                запись.getJSONArray("top5").getInt(0),
                результат.matches.first().classIdx,
            )
        }
    }

    @Test
    fun детектор_совпадает_по_iou() {
        val записи = эталон.getJSONArray("frame")
        for (индекс in 0 until записи.length()) {
            val запись = записи.getJSONObject(индекс)
            val найденные = зрение.detectBitmapForTest(картинка(запись.getString("png")))
            assertTrue("кадр $индекс: детектор ничего не нашёл", найденные.isNotEmpty())
            val ожидаемый = боксИз(запись, "detector_box")
            val лучший = найденные.maxByOrNull { Boxes.iou(it, ожидаемый) }!!
            val iou = Boxes.iou(лучший, ожидаемый)
            assertTrue("кадр $индекс: IoU $iou < 0.9", iou >= 0.9f)
        }
    }
}
