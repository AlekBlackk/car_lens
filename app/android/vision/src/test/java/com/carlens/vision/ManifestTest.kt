package com.carlens.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class ManifestTest {

    // Образец повторяет то, что пишет `scripts/stage_assets.py:99-116`: `num_classes`
    // числом, а не строкой, и два ключа про детектор в `provenance`. Разбор обязан
    // проверяться на той форме, которая приезжает на устройство.
    private val образец = """
    {
      "files": {
        "classifier": {"name": "classifier_int8.tflite", "size": 12, "sha256": "${"aa".repeat(32)}"},
        "detector":   {"name": "detector_fp32.tflite",  "size": 34, "sha256": "${"bb".repeat(32)}"},
        "index":      {"name": "catalog_index.bin",     "size": 56, "sha256": "${"cc".repeat(32)}"},
        "meta":       {"name": "catalog_meta.json",     "size": 78, "sha256": "${"dd".repeat(32)}"}
      },
      "provenance": {
        "generated_at": "2026-08-19T10:00:00+00:00",
        "checkpoint_sha256": "${"ee".repeat(32)}",
        "git_sha": "498c580",
        "num_classes": 171,
        "index_source": "checkpoint",
        "detector_weights_sha256": "${"11".repeat(32)}",
        "detector_graph_sha256": "${"22".repeat(32)}"
      },
      "classifier": {
        "img_size": 320, "embedding_dim": 256,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
      },
      "detector": {
        "imgsz": 416, "input_dtype": "float32", "input_layout": "NHWC",
        "channel_order": "RGB", "input_range": "unit", "letterbox_pad_value": 114,
        "output_layout": "CN", "box_format": "xywh", "coord_units": "normalized",
        "num_classes": 80, "vehicle_classes": [2, 5, 7],
        "score_layout": "classes_only", "max_det": 100
      }
    }
    """.trimIndent().toByteArray(Charsets.UTF_8)

    private fun испортить(что: String, чем: String): ByteArray =
        String(образец, Charsets.UTF_8).replace(что, чем).toByteArray(Charsets.UTF_8)

    @Test
    fun `манифест разбирается целиком`() {
        val манифест = ModelsManifest.parse(образец)
        assertEquals("classifier_int8.tflite", манифест.files.getValue("classifier").name)
        assertEquals(320, манифест.classifier.imgSize)
        assertEquals(256, манифест.classifier.embeddingDim)
        assertEquals("normalized", манифест.detector.coordUnits)
        assertEquals(listOf(2, 5, 7), манифест.detector.vehicleClasses)
    }

    @Test
    fun `provenance читается независимо от типа значения`() {
        val provenance = ModelsManifest.parse(образец).provenance
        assertEquals("171", provenance.getValue("num_classes"))
        assertEquals("498c580", provenance.getValue("git_sha"))
        assertEquals("22".repeat(32), provenance.getValue("detector_graph_sha256"))
    }

    @Test
    fun `спецификация детектора собирается из контракта`() {
        val спек = ModelsManifest.parse(образец).detectorSpec()
        assertEquals(416, спек.size)
        assertEquals(TensorLayout.NHWC, спек.layout)
        assertEquals(ChannelOrder.RGB, спек.channels)
        assertTrue(спек.scaling is Scaling.Unit01)
    }

    @Test
    fun `спецификация классификатора берёт mean и std`() {
        val спек = ModelsManifest.parse(образец).classifierSpec()
        assertEquals(320, спек.size)
        val нормализация = спек.scaling as Scaling.MeanStd
        assertEquals(0.485f, нормализация.mean[0], 1e-6f)
        assertEquals(0.225f, нормализация.std[2], 1e-6f)
    }

    @Test
    fun `неизвестное значение input_range роняет с самим значением`() {
        val испорченный = испортить("\"input_range\": \"unit\"", "\"input_range\": \"whatever\"")
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            ModelsManifest.parse(испорченный).detectorSpec()
        }
        assertTrue(отказ.message!!.contains("whatever"))
    }

    @Test
    fun `режим mean_std у детектора роняет отдельным сообщением`() {
        val испорченный = испортить("\"input_range\": \"unit\"", "\"input_range\": \"mean_std\"")
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            ModelsManifest.parse(испорченный).detectorSpec()
        }
        assertTrue(отказ.message!!.contains("mean_std"))
        assertTrue(отказ.message!!.contains("255"))
    }

    @Test
    fun `нефлотовый вход детектора роняет с самим типом`() {
        // Квантованный детектор потребовал бы других границ тензора: молча скормить
        // ему float означает мусор на входе без единого отказа.
        val испорченный = испортить("\"input_dtype\": \"float32\"", "\"input_dtype\": \"uint8\"")
        val отказ = assertThrows(IllegalArgumentException::class.java) {
            ModelsManifest.parse(испорченный).detectorSpec()
        }
        assertTrue(отказ.message!!.contains("uint8"))
    }

    @Test
    fun `неизвестные раскладка и порядок каналов роняют со своим значением`() {
        val поРаскладке = assertThrows(IllegalArgumentException::class.java) {
            ModelsManifest.parse(испортить("\"input_layout\": \"NHWC\"", "\"input_layout\": \"HWCN\""))
                .detectorSpec()
        }
        assertTrue(поРаскладке.message!!.contains("HWCN"))
        val поКаналам = assertThrows(IllegalArgumentException::class.java) {
            ModelsManifest.parse(испортить("\"channel_order\": \"RGB\"", "\"channel_order\": \"GBR\""))
                .detectorSpec()
        }
        assertTrue(поКаналам.message!!.contains("GBR"))
    }

    @Test
    fun `отсутствие группы detector роняет`() {
        val урезанный = String(образец, Charsets.UTF_8)
            .replaceAfter("\"detector\": {", "}}")
            .toByteArray(Charsets.UTF_8)
        assertThrows(Exception::class.java) { ModelsManifest.parse(урезанный) }
    }

    @Test
    fun `пропущенное поле контракта детектора роняет, а не подставляет ноль`() {
        val испорченный = испортить("\"max_det\": 100", "\"maxDet\": 100")
        assertThrows(Exception::class.java) { ModelsManifest.parse(испорченный) }
    }

    @Test
    fun `ассет с чужим sha256 роняет и называет оба хеша`() {
        val данные = "модель".toByteArray(Charsets.UTF_8)
        val ожидаемый = "ff".repeat(32)
        val отказ = assertThrows(AssetMismatchException::class.java) {
            Assets.verified(данные, name = "detector_fp32.tflite", expectedSha256 = ожидаемый)
        }
        assertTrue(отказ.message!!.contains(Embeddings.sha256(данные)))
        assertTrue(отказ.message!!.contains(ожидаемый))
        assertTrue(отказ.message!!.contains("detector_fp32.tflite"))
    }

    @Test
    fun `ассет со своим sha256 проходит, пустой эталон пропускает сверку`() {
        val данные = "модель".toByteArray(Charsets.UTF_8)
        assertTrue(
            Assets.verified(данные, "x", Embeddings.sha256(данные)).contentEquals(данные)
        )
        assertTrue(Assets.verified(данные, "x", expectedSha256 = null).contentEquals(данные))
    }

    @Test
    fun `файлы манифеста дают имя, размер и хеш`() {
        val индекс = ModelsManifest.parse(образец).files.getValue("index")
        assertEquals("catalog_index.bin", индекс.name)
        assertEquals(56L, индекс.size)
        assertEquals("cc".repeat(32), индекс.sha256)
        assertNull(ModelsManifest.parse(образец).files["чего-нет"])
    }
}
