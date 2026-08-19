package com.carlens.vision

import android.content.res.AssetManager
import android.media.Image
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class EmbedResult(val embedding: FloatArray, val matches: List<Match>) {
    override fun equals(other: Any?): Boolean =
        this === other || (
            other is EmbedResult &&
                embedding.contentEquals(other.embedding) &&
                matches == other.matches
            )

    override fun hashCode(): Int = 31 * embedding.contentHashCode() + matches.hashCode()
}

data class IndexInfo(val numClasses: Int, val embeddingDim: Int, val centroidsPerClass: Int)

/**
 * Что грузить и что сверять.
 *
 * `expected` перекрывает хеши манифеста: манифест лежит в том же APK, что и модели, и
 * сам себя подтвердить не может — внешний эталон нужен тестам и вызывающему, который
 * знает набор заранее.
 */
data class AssetSpec(
    val manifestName: String = "models_manifest.json",
    val expected: Map<String, String> = emptyMap(),
)

/**
 * Фасад распознавания: детектор, классификатор и индекс за одной ручкой.
 *
 * Поднимаются вместе и падают вместе: по отдельности они бессмысленны, а расходятся
 * молча — классификатор от одного прогона с индексом от другого даёт уверенные
 * неправильные подписи (§10 дизайна).
 */
class CarVision private constructor(
    private val index: CatalogIndex,
    private val classifier: Classifier,
    private val detector: Detector,
) : AutoCloseable {

    private var закрыт = false

    val indexInfo: IndexInfo = IndexInfo(
        numClasses = index.header.numClasses,
        embeddingDim = index.header.embeddingDim,
        centroidsPerClass = index.header.centroidsPerClass,
    )

    val delegateMode: DelegateMode get() = detector.delegateMode

    val classes: List<CatalogClass> get() = index.classes

    /** Рамки транспорта в пикселях выпрямленного кадра. Кадр не закрывается и не хранится. */
    fun detect(image: Image, rotationDegrees: Int): List<Box> {
        проверитьЖивость()
        return detector.detect(кадр(image, rotationDegrees))
    }

    /** Эмбеддинг каждого бокса и его top-K по индексу. */
    fun embedAndMatch(
        image: Image,
        rotationDegrees: Int,
        boxes: List<Box>,
        pad: Float = 0.08f,
        topK: Int = 5,
    ): List<EmbedResult> {
        проверитьЖивость()
        val frame = кадр(image, rotationDegrees)
        return boxes.map { бокс ->
            // `embed` возвращает свежий массив на каждый бокс: общий буфер на список
            // затирал бы предыдущий результат, а наружу уехали бы одинаковые эмбеддинги.
            val эмбеддинг = classifier.embed(frame, бокс, pad)
            EmbedResult(эмбеддинг, index.knn(эмбеддинг, topK))
        }
    }

    /** Совпадения для готового вектора: нормализацию берёт на себя. */
    fun matchEmbedding(embedding: FloatArray, topK: Int = 5): List<Match> {
        проверитьЖивость()
        val нормированный = Embeddings.l2NormalizedCopy(embedding, indexInfo.embeddingDim)
        return index.knn(нормированный, topK)
    }

    /**
     * Кадр поверх плоскостей `Image` без копирования буферов.
     *
     * Плагин не закрывает `Image` и не держит ссылку после возврата (§5.2 дизайна):
     * владелец кадра — вызывающий, а буферы живут ровно до его `close()`.
     */
    private fun кадр(image: Image, rotationDegrees: Int): YuvFrame {
        val яркость = image.planes[0]
        val синева = image.planes[1]
        val краснота = image.planes[2]
        check(синева.rowStride == краснота.rowStride && синева.pixelStride == краснота.pixelStride) {
            "плоскости хромы с разными шагами: U ${синева.rowStride}/${синева.pixelStride}, " +
                "V ${краснота.rowStride}/${краснота.pixelStride}"
        }
        return YuvFrame(
            width = image.width,
            height = image.height,
            rotationDegrees = rotationDegrees,
            y = яркость.buffer,
            yRowStride = яркость.rowStride,
            yPixelStride = яркость.pixelStride,
            u = синева.buffer,
            v = краснота.buffer,
            uvRowStride = синева.rowStride,
            uvPixelStride = синева.pixelStride,
        )
    }

    private fun проверитьЖивость() {
        check(!закрыт) { "CarVision уже закрыт вызовом close()" }
    }

    /** Идемпотентно: повторный `close` — не ошибка, тесты и жизненный цикл Android зовут его дважды. */
    override fun close() {
        if (закрыт) return
        закрыт = true
        classifier.close()
        detector.close()
    }

    companion object {

        fun load(assets: AssetManager, spec: AssetSpec = AssetSpec()): CarVision {
            val манифест = ModelsManifest.parse(
                assets.open(spec.manifestName).use { it.readBytes() }
            )
            val индексныеБайты = прочитать(assets, манифест, spec, "index")
            val мета = прочитать(assets, манифест, spec, "meta")
            val индекс = CatalogIndex.parse(индексныеБайты, мета, Embeddings::sha256)

            val классификатор = Classifier(
                model = прямойБуфер(прочитать(assets, манифест, spec, "classifier")),
                spec = манифест.classifierSpec(),
                expectedEmbeddingDim = индекс.header.embeddingDim,
                options = org.tensorflow.lite.Interpreter.Options().setNumThreads(2),
            )
            val детектор = runCatching {
                Detector(
                    model = прямойБуфер(прочитать(assets, манифест, spec, "detector")),
                    contract = манифест.detector,
                    spec = манифест.detectorSpec(),
                )
            }.getOrElse { отказ ->
                классификатор.close()
                throw отказ
            }
            return CarVision(индекс, классификатор, детектор)
        }

        private fun прочитать(
            assets: AssetManager,
            манифест: ModelsManifest,
            spec: AssetSpec,
            ключ: String,
        ): ByteArray {
            val запись = манифест.files[ключ]
                ?: throw AssetMismatchException("в манифесте нет файла \"$ключ\"")
            val данные = Assets.readVerified(
                assets, запись.name, spec.expected[ключ] ?: запись.sha256,
            )
            if (данные.size.toLong() != запись.size) {
                throw AssetMismatchException(
                    "размер ассета ${запись.name}: ${данные.size} против ${запись.size} в манифесте"
                )
            }
            return данные
        }

        /** Интерпретатор требует прямой буфер: обычный `ByteArray` он не примет. */
        private fun прямойБуфер(данные: ByteArray): ByteBuffer =
            ByteBuffer.allocateDirect(данные.size).order(ByteOrder.nativeOrder()).apply {
                put(данные)
                rewind()
            }
    }
}
