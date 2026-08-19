package com.carlens.vision

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive

@Serializable
data class FileEntry(val name: String, val size: Long, val sha256: String)

@Serializable
data class ClassifierContract(
    @SerialName("img_size") val imgSize: Int,
    val mean: FloatArray,
    val std: FloatArray,
    @SerialName("embedding_dim") val embeddingDim: Int,
) {
    // Содержательные `equals`/`hashCode`: у data class с массивами они сравнивают ссылки.
    override fun equals(other: Any?): Boolean =
        this === other || (
            other is ClassifierContract &&
                imgSize == other.imgSize &&
                embeddingDim == other.embeddingDim &&
                mean.contentEquals(other.mean) &&
                std.contentEquals(other.std)
            )

    override fun hashCode(): Int {
        var хеш = imgSize
        хеш = 31 * хеш + embeddingDim
        хеш = 31 * хеш + mean.contentHashCode()
        return 31 * хеш + std.contentHashCode()
    }
}

/** Контракт экспортированного детектора (§7.1 дизайна), поля — как в `export/detector_contract.py`. */
@Serializable
data class DetectorContract(
    val imgsz: Int,
    @SerialName("input_dtype") val inputDtype: String,
    @SerialName("input_layout") val inputLayout: String,
    @SerialName("channel_order") val channelOrder: String,
    @SerialName("input_range") val inputRange: String,
    @SerialName("letterbox_pad_value") val letterboxPadValue: Int,
    @SerialName("output_layout") val outputLayout: String,
    @SerialName("box_format") val boxFormat: String,
    @SerialName("coord_units") val coordUnits: String,
    @SerialName("num_classes") val numClasses: Int,
    @SerialName("vehicle_classes") val vehicleClasses: List<Int>,
    @SerialName("score_layout") val scoreLayout: String,
    @SerialName("max_det") val maxDet: Int,
)

@Serializable
private data class RawManifest(
    val files: Map<String, FileEntry>,
    // Значения читаются как примитивы, а не как строки: `stage_assets.py` пишет
    // `num_classes` числом, и `Map<String, String>` уронил бы разбор настоящего
    // манифеста, оставив тесты на выдуманном образце зелёными.
    val provenance: Map<String, JsonPrimitive>,
    val classifier: ClassifierContract,
    val detector: DetectorContract,
)

data class ModelsManifest(
    val files: Map<String, FileEntry>,
    val provenance: Map<String, String>,
    val classifier: ClassifierContract,
    val detector: DetectorContract,
) {
    /** Спецификация входного тензора детектора по его контракту. */
    fun detectorSpec(): TensorSpec {
        require(detector.inputDtype == "float32") {
            "input_dtype \"${detector.inputDtype}\": веха работает с fp32-входом детектора"
        }
        val нормализация = when (detector.inputRange) {
            "unit" -> Scaling.Unit01
            "mean_std" -> throw IllegalArgumentException(
                "input_range \"mean_std\" у детектора вехой не поддержан: ultralytics делит " +
                    "вход на 255 без вычитания среднего"
            )
            else -> throw IllegalArgumentException(
                "неизвестный input_range \"${detector.inputRange}\""
            )
        }
        return TensorSpec(
            size = detector.imgsz,
            layout = раскладка(detector.inputLayout),
            channels = каналы(detector.channelOrder),
            scaling = нормализация,
        )
    }

    /**
     * Спецификация входного тензора классификатора.
     *
     * Раскладка и порядок каналов взяты NHWC/RGB, потому что в манифесте их нет:
     * `stage_assets.py` переносит из отчёта экспорта только `img_size`, `mean`, `std` и
     * `embedding_dim`, хотя конвертер пишет раскладку графа в `export_report.json`
     * (`export/to_tflite.py:232`). Расхождение поймает задача 9 при создании
     * интерпретатора — форма входа графа известна там, — но правильнее перенести поле в
     * манифест; это долг вехи, а не решение.
     */
    fun classifierSpec(): TensorSpec = TensorSpec(
        size = classifier.imgSize,
        layout = TensorLayout.NHWC,
        channels = ChannelOrder.RGB,
        scaling = Scaling.MeanStd(mean = classifier.mean, std = classifier.std),
    )

    private fun раскладка(значение: String): TensorLayout = when (значение) {
        "NHWC" -> TensorLayout.NHWC
        "NCHW" -> TensorLayout.NCHW
        else -> throw IllegalArgumentException("неизвестный input_layout \"$значение\"")
    }

    private fun каналы(значение: String): ChannelOrder = when (значение) {
        "RGB" -> ChannelOrder.RGB
        "BGR" -> ChannelOrder.BGR
        else -> throw IllegalArgumentException("неизвестный channel_order \"$значение\"")
    }

    companion object {
        // Незнакомые ключи пропускаются: `provenance` и контракты пополняются на
        // питоновской стороне, и новое поле не повод не запуститься. Значений по
        // умолчанию у полей нет: пропуск обязательного поля обязан ронять разбор.
        private val разборщик = Json { ignoreUnknownKeys = true }

        fun parse(json: ByteArray): ModelsManifest {
            val сырой = разборщик.decodeFromString<RawManifest>(String(json, Charsets.UTF_8))
            return ModelsManifest(
                files = сырой.files,
                provenance = сырой.provenance.mapValues { (_, значение) -> значение.content },
                classifier = сырой.classifier,
                detector = сырой.detector,
            )
        }
    }
}
