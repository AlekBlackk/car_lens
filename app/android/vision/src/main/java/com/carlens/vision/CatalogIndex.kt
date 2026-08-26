package com.carlens.vision

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Версия формата `catalog_index.bin`, поддерживаемая этой сборкой (§13.2 спеки). */
const val INDEX_FORMAT_VERSION: Int = 1

private const val MAGIC = "CLIX"
private const val HEADER_SIZE = 64
private const val DTYPE_FLOAT16 = 0
private const val DTYPE_FLOAT32 = 1

class IndexFormatException(message: String) : IllegalArgumentException(message)

data class IndexHeader(
    val formatVersion: Int,
    val embeddingDim: Int,
    val numClasses: Int,
    val centroidsPerClass: Int,
    val dtype: Int,
    val normalized: Boolean,
    val metaSha256: String,
)

/**
 * Запись каталога описывает класс, а не модель (§13.2): состав группы лежит в `members`,
 * а `brand` пуст у разнобрендовой группы — «Geely Coolray / Belgee X50» одной маркой не
 * назвать. Поля `idx` нет намеренно: индексом класса служит позиция в массиве `classes`.
 *
 * Значений по умолчанию ни у одного поля нет: мету пишет наш же `pipeline/index_format.py`,
 * и отсутствие поля означает не «пустое значение», а мету другого формата — её надо
 * отвергать, а не додумывать.
 */
@Serializable
data class CatalogClass(
    @SerialName("class_id") val classId: String,
    val members: List<String>,
    val brand: String?,
    val display: String,
)

@Serializable
private data class CatalogMeta(val classes: List<CatalogClass> = emptyList())

data class Match(val classIdx: Int, val sim: Float)

class CatalogIndex(
    val header: IndexHeader,
    val classes: List<CatalogClass>,
    private val body: FloatArray,
) {
    fun centroid(classIdx: Int, k: Int): FloatArray {
        if (classIdx !in 0 until header.numClasses) {
            throw IndexFormatException("класс $classIdx вне 0..${header.numClasses - 1}")
        }
        if (k !in 0 until header.centroidsPerClass) {
            throw IndexFormatException("центроид $k вне 0..${header.centroidsPerClass - 1}")
        }
        val смещение = (classIdx * header.centroidsPerClass + k) * header.embeddingDim
        return body.copyOfRange(смещение, смещение + header.embeddingDim)
    }

    /** Вектор обязан быть нормализован: нормализует `Embeddings.l2NormalizedCopy`. */
    fun knn(embedding: FloatArray, topK: Int): List<Match> {
        if (embedding.size != header.embeddingDim) {
            throw IndexFormatException(
                "размерность ${embedding.size} против ${header.embeddingDim} в индексе"
            )
        }
        // Отбор ниже читает `лучшиеСходства[размер - 1]`, поэтому бессмысленный `topK`
        // обязан отвергаться здесь, а не выходить за границы массива из глубины цикла.
        if (topK < 1) throw IndexFormatException("topK $topK: ответ не может быть пустым")
        val размер = minOf(topK, header.numClasses)
        val лучшиеКлассы = IntArray(размер) { -1 }
        val лучшиеСходства = FloatArray(размер) { Float.NEGATIVE_INFINITY }
        for (класс in 0 until header.numClasses) {
            var максимум = Float.NEGATIVE_INFINITY
            for (центроид in 0 until header.centroidsPerClass) {
                var произведение = 0f
                val смещение = (класс * header.centroidsPerClass + центроид) * header.embeddingDim
                for (измерение in 0 until header.embeddingDim) {
                    произведение += body[смещение + измерение] * embedding[измерение]
                }
                if (произведение > максимум) максимум = произведение
            }
            // Вставка в отсортированный массив на topK элементов: при K=5 это дешевле
            // сортировки всех классов и ничего не аллоцирует в горячем пути.
            if (максимум <= лучшиеСходства[размер - 1]) continue
            var позиция = размер - 1
            while (позиция > 0 && лучшиеСходства[позиция - 1] < максимум) {
                лучшиеСходства[позиция] = лучшиеСходства[позиция - 1]
                лучшиеКлассы[позиция] = лучшиеКлассы[позиция - 1]
                позиция -= 1
            }
            лучшиеСходства[позиция] = максимум
            лучшиеКлассы[позиция] = класс
        }
        return List(размер) { индекс -> Match(лучшиеКлассы[индекс], лучшиеСходства[индекс]) }
    }

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        fun parse(bin: ByteArray, meta: ByteArray, sha256: (ByteArray) -> String): CatalogIndex {
            val заголовок = разобратьЗаголовок(bin)
            val значений =
                заголовок.numClasses * заголовок.centroidsPerClass * заголовок.embeddingDim
            val ожидаемая = HEADER_SIZE + значений * 4
            if (bin.size != ожидаемая) {
                throw IndexFormatException("длина ${bin.size} против $ожидаемая по заголовку")
            }
            val фактический = sha256(meta)
            if (фактический != заголовок.metaSha256) {
                throw IndexFormatException(
                    "sha256 меты $фактический против ${заголовок.metaSha256} в заголовке"
                )
            }
            val классы = разобратьМету(meta)
            if (классы.size != заголовок.numClasses) {
                throw IndexFormatException(
                    "classes меты: ${классы.size} против ${заголовок.numClasses} в заголовке"
                )
            }
            val тело = FloatArray(значений)
            ByteBuffer.wrap(bin, HEADER_SIZE, значений * 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()
                .get(тело)
            return CatalogIndex(заголовок, классы, тело)
        }

        /**
         * Мета старого формата, где класс равен модели, роняет `kotlinx.serialization`
         * собственным `MissingFieldException` — из ассетов APK его видно как «что-то
         * сломалось в сериализации». Единственный вызывающий `parse` (`CarVision.load`)
         * разбирает ошибки формата индекса, поэтому и эта приходит к нему тем же типом,
         * с текстом kotlinx внутри: он называет пропавшее поле.
         */
        private fun разобратьМету(meta: ByteArray): List<CatalogClass> = try {
            json.decodeFromString<CatalogMeta>(String(meta, Charsets.UTF_8)).classes
        } catch (ошибка: SerializationException) {
            throw IndexFormatException("catalog_meta.json не по §13.2: ${ошибка.message}")
        }

        private fun разобратьЗаголовок(bin: ByteArray): IndexHeader {
            if (bin.size < HEADER_SIZE) {
                throw IndexFormatException("файл короче заголовка (${bin.size} < $HEADER_SIZE)")
            }
            val буфер = ByteBuffer.wrap(bin).order(ByteOrder.LITTLE_ENDIAN)
            val magic = String(bin, 0, 4, Charsets.US_ASCII)
            if (magic != MAGIC) throw IndexFormatException("magic \"$magic\" вместо \"$MAGIC\"")
            val версия = буфер.getShort(4).toInt() and 0xFFFF
            if (версия != INDEX_FORMAT_VERSION) {
                throw IndexFormatException(
                    "format_version $версия, поддерживается $INDEX_FORMAT_VERSION"
                )
            }
            val dtype = bin[16].toInt() and 0xFF
            if (dtype == DTYPE_FLOAT16) {
                throw IndexFormatException(
                    "float16 не поддерживается: §13.2 предписывает float32 в v1"
                )
            }
            if (dtype != DTYPE_FLOAT32) throw IndexFormatException("неизвестный dtype $dtype")
            val reservedНенулевой =
                (буфер.getShort(18).toInt() != 0) || (52 until 64).any { bin[it].toInt() != 0 }
            if (reservedНенулевой) {
                throw IndexFormatException("reserved не нулевой — файл писала другая версия формата")
            }
            return IndexHeader(
                formatVersion = версия,
                embeddingDim = буфер.getShort(6).toInt() and 0xFFFF,
                numClasses = буфер.getInt(8),
                centroidsPerClass = буфер.getInt(12),
                dtype = dtype,
                normalized = (bin[17].toInt() and 0xFF) == 1,
                metaSha256 = (20 until 52).joinToString("") { "%02x".format(bin[it]) },
            )
        }
    }
}
