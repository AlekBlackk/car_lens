package com.carlens.vision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.system.Os
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipEntry
import java.util.zip.ZipFile

/**
 * Эмбеддинги одних и тех же кропов на CPU и на GPU-делегате классификатора (M4, латентность).
 *
 * Тест ничего не утверждает про точность: GPU считает активации в fp16, и косинус к CPU по
 * определению ниже 0.9999. Он пишет два файла эмбеддингов, а решения (top-1, принять или
 * отказать в точке телефона) считает `training.eval_device_agreement report` на тех же
 * порогах и индексе, что и приёмка. Запускается вручную: кропы (несколько сотен мегабайт)
 * кладёт `training.eval_device_agreement pack`, а не сборка.
 *
 * Обмен через `getExternalFilesDir(null)/agree/`: `crops.zip` на входе, `emb_cpu.f32` и
 * `emb_gpu.f32` (float32, little-endian, `[кроп, dim]`, L2-нормированные) на выходе. Кропы
 * идут в тот же путь, что и в `ParityTest` уровня A: `embedBitmapForTest`.
 */
class GpuAgreementTest {

    @Test
    fun эмбеддинги_кропов_на_cpu_и_gpu() {
        val контекст = InstrumentationRegistry.getInstrumentation().context
        val папка = File(
            InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null),
            "agree",
        )
        println("ПАПКА_СОГЛАСИЯ ${папка.absolutePath}")
        val архив = File(папка, "crops.zip")
        assumeTrue("нет $архив: сначала `training.eval_device_agreement pack` и adb push", архив.exists())

        val наCpu = CarVision.load(контекст.assets, AssetSpec(classifierGpu = false))
        val наGpu = CarVision.load(контекст.assets, AssetSpec(classifierGpu = true))
        try {
            assertEquals(DelegateMode.CPU, наCpu.classifierDelegateMode)
            // Без GPU согласие превратилось бы в сравнение CPU с самим собой и «прошло» бы молча.
            assertEquals(DelegateMode.GPU, наGpu.classifierDelegateMode)
            val измерение = наCpu.indexInfo.embeddingDim
            ZipFile(архив).use { zip ->
                // Порядок записей в архиве — порядок строк `order.csv`; имена отсортированы для надёжности.
                val записи = zip.entries().asSequence().sortedBy { it.name }.toList()
                val выходCpu = File(папка, "emb_cpu.f32").outputStream().buffered()
                val выходGpu = File(папка, "emb_gpu.f32").outputStream().buffered()
                val байты = ByteBuffer.allocate(измерение * 4).order(ByteOrder.LITTLE_ENDIAN)
                fun записать(вектор: FloatArray, куда: java.io.OutputStream) {
                    байты.clear()
                    байты.asFloatBuffer().put(вектор)
                    куда.write(байты.array())
                }
                записи.forEachIndexed { номер, запись ->
                    val картинка = декодировать(zip, запись)
                    записать(наCpu.embedBitmapForTest(картинка), выходCpu)
                    записать(наGpu.embedBitmapForTest(картинка), выходGpu)
                    картинка.recycle()
                    if (номер % 500 == 0) println("СОГЛАСИЕ $номер/${записи.size}")
                }
                выходCpu.close()
                выходGpu.close()
                открытьДляЧтения(папка)
                println("СОГЛАСИЕ готово ${записи.size} кропов, dim=$измерение")
            }
        } finally {
            наCpu.close()
            наGpu.close()
        }
    }

    /**
     * Результаты создаёт приложение с правами 660, а `adb pull` идёт от имени shell и их не читает
     * (`run-as` тестовому APK недоступен: он не debuggable). Доступ выдаёт сам владелец файлов.
     */
    private fun открытьДляЧтения(папка: File) {
        for (имя in listOf("emb_cpu.f32", "emb_gpu.f32")) {
            val файл = File(папка, имя)
            if (файл.exists()) Os.chmod(файл.path, 420) // 0644
        }
    }

    /** Повторно открыть результаты прошлого прогона, не считая их заново. */
    @Test
    fun открыть_результаты_для_adb_pull() {
        val папка = File(
            InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null),
            "agree",
        )
        assumeTrue("нет $папка", папка.exists())
        открытьДляЧтения(папка)
    }

    /**
     * Декодирование с понижением, если длинная сторона больше `ПРЕДЕЛ_СТОРОНЫ`: часть исходников
     * в наборе — снимки в десятки мегапикселей (до 12455×8283), которые `BitmapFactory` целиком
     * не берёт. Понижение одинаково для CPU- и GPU-пути, на их согласие не влияет.
     */
    private fun декодировать(zip: ZipFile, запись: ZipEntry): Bitmap {
        val границы = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        zip.getInputStream(запись).use { BitmapFactory.decodeStream(it, null, границы) }
        var понижение = 1
        while (maxOf(границы.outWidth, границы.outHeight) / понижение > ПРЕДЕЛ_СТОРОНЫ) понижение *= 2
        val параметры = BitmapFactory.Options().apply { inSampleSize = понижение }
        return zip.getInputStream(запись).use { BitmapFactory.decodeStream(it, null, параметры) }
            ?: error("не читается ${запись.name} (${границы.outWidth}×${границы.outHeight}, понижение $понижение)")
    }

    private companion object {
        const val ПРЕДЕЛ_СТОРОНЫ = 4096
    }
}
