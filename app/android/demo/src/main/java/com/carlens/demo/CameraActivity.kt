package com.carlens.demo

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.carlens.vision.AssetSpec
import com.carlens.vision.CarVision
import com.carlens.vision.Match
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Демонстрация распознавания: превью камеры, рамка и подпись.
 *
 * Трекера здесь нет. Он живёт в JS и пишется на M-1.6, и подделка его тут означала бы
 * выбросить работу дважды: `embedAndMatch` просто зовётся не чаще двух раз в секунду,
 * а рамки берутся от детектора каждый кадр.
 */
class CameraActivity : ComponentActivity() {

    private lateinit var превью: PreviewView
    private lateinit var оверлей: OverlayView
    private lateinit var сообщение: TextView
    private lateinit var времена: TextView

    // Один исполнитель на загрузку и на анализ: вызовы `CarVision` синхронны и не
    // потокобезопасны (§5.2 дизайна), а общий поток делает это гарантией, а не
    // договорённостью.
    private lateinit var исполнитель: ExecutorService

    @Volatile
    private var зрение: CarVision? = null

    private var анализ: ImageAnalysis? = null

    private var последнийЭмбеддинг = 0L

    private val запросРазрешения =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { дали ->
            if (дали) поднятьЗрениеИКамеру() else показать(getString(R.string.разрешение_нужно))
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)
        превью = findViewById(R.id.превью)
        оверлей = findViewById(R.id.оверлей)
        сообщение = findViewById(R.id.сообщение)
        времена = findViewById(R.id.времена)
        оверлей.приСостоянии = { строка -> времена.text = строка }
        исполнитель = Executors.newSingleThreadExecutor()

        val естьРазрешение = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        if (естьРазрешение) поднятьЗрениеИКамеру() else запросРазрешения.launch(Manifest.permission.CAMERA)
    }

    /**
     * Модели грузятся на том же потоке, что и анализ, и занимают секунды: держать этим
     * главный поток нельзя, а стартовать камеру раньше моделей незачем — кадры всё равно
     * некому разбирать.
     */
    private fun поднятьЗрениеИКамеру() {
        исполнитель.execute {
            val результат = runCatching {
                CarVision.load(assets, AssetSpec(expected = ожидаемыеХеши()))
            }
            runOnUiThread {
                результат.onSuccess { поднятое ->
                    зрение = поднятое
                    стартоватьКамеру()
                }.onFailure { отказ ->
                    Log.e(ТЕГ, "модели не загрузились", отказ)
                    показать(getString(R.string.модели_не_загрузились, отказ.message ?: ""))
                }
            }
        }
    }

    /**
     * Ожидаемые суммы вшиты в APK на этапе сборки (`BuildConfig`).
     *
     * Это сверка «те ли модели попали в APK, что раскладывались», а не защита от
     * подмены: эталон лежит в том же артефакте, что и проверяемый файл (§10 дизайна).
     */
    private fun ожидаемыеХеши(): Map<String, String> = mapOf(
        "classifier" to BuildConfig.SHA_CLASSIFIER,
        "detector" to BuildConfig.SHA_DETECTOR,
        "index" to BuildConfig.SHA_INDEX,
        "meta" to BuildConfig.SHA_META,
    ).filterValues { it.isNotEmpty() }

    private fun стартоватьКамеру() {
        val будущий = ProcessCameraProvider.getInstance(this)
        будущий.addListener({
            val провайдер = runCatching { будущий.get() }.getOrElse { отказ ->
                Log.e(ТЕГ, "камера не поднялась", отказ)
                показать(getString(R.string.камера_не_поднялась, отказ.message ?: ""))
                return@addListener
            }
            val превьюВариант = Preview.Builder().build().also {
                it.surfaceProvider = превью.surfaceProvider
            }
            val текущийАнализ = ImageAnalysis.Builder()
                // Разбирается только последний кадр: очередь кадров на медленном
                // устройстве превращается в растущую задержку между тем, что на экране,
                // и тем, что разобрано.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetRotation(display.rotation)
                .build()
                .also { it.setAnalyzer(исполнитель, анализатор) }
            анализ = текущийАнализ
            провайдер.unbindAll()
            runCatching {
                провайдер.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, превьюВариант, текущийАнализ,
                )
            }.onFailure { отказ ->
                Log.e(ТЕГ, "камера не поднялась", отказ)
                показать(getString(R.string.камера_не_поднялась, отказ.message ?: ""))
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private val анализатор = ImageAnalysis.Analyzer { кадр ->
        try {
            val текущее = зрение ?: return@Analyzer
            val изображение = кадр.image ?: return@Analyzer
            val угол = кадр.imageInfo.rotationDegrees
            val начало = SystemClock.elapsedRealtimeNanos()
            val боксы = текущее.detect(изображение, угол)
            val послеДетектора = SystemClock.elapsedRealtimeNanos()
            val результаты =
                if (боксы.isNotEmpty() && пораСчитать()) {
                    текущее.embedAndMatch(изображение, угол, боксы.take(2), pad = 0.08f, topK = 5)
                } else {
                    emptyList()
                }
            val конец = SystemClock.elapsedRealtimeNanos()
            val подписи = результаты.map { подпись(текущее, it.matches.first()) }
            val выпрямленный = if (угол == 90 || угол == 270) {
                изображение.height to изображение.width
            } else {
                изображение.width to изображение.height
            }
            оверлей.post {
                оверлей.показать(
                    боксы = боксы,
                    подписи = подписи,
                    кадрШирина = выпрямленный.first,
                    кадрВысота = выпрямленный.second,
                    времена = Времена(
                        детектор = (послеДетектора - начало) / 1_000_000f,
                        классификатор = if (результаты.isEmpty()) {
                            0f
                        } else {
                            (конец - послеДетектора) / 1_000_000f
                        },
                    ),
                    делегат = текущее.delegateMode,
                )
            }
        } catch (отказ: RuntimeException) {
            Log.e(ТЕГ, "кадр не разобран", отказ)
        } finally {
            // Всегда: плагин кадр не закрывает и ссылку на него не хранит (§5.2 дизайна),
            // а незакрытый кадр останавливает поток анализа целиком.
            кадр.close()
        }
    }

    /** Эмбеддинг не чаще двух раз в секунду: подпись меняется медленно, а кадр дорогой. */
    private fun пораСчитать(): Boolean {
        val сейчас = SystemClock.elapsedRealtime()
        if (сейчас - последнийЭмбеддинг < ПЕРИОД_ЭМБЕДДИНГА_МС) return false
        последнийЭмбеддинг = сейчас
        return true
    }

    private fun подпись(зрение: CarVision, совпадение: Match): String {
        val класс = зрение.classes[совпадение.classIdx]
        return "${класс.display} · ${(совпадение.sim * 100).roundToInt()}%"
    }

    private fun показать(текст: String) {
        сообщение.text = текст
        сообщение.visibility = TextView.VISIBLE
    }

    /**
     * Поворот телефона: `Activity` не пересоздаётся (иначе модели грузились бы заново
     * по три секунды на каждый поворот), поэтому новую ориентацию анализу надо сообщить
     * руками. Иначе `rotationDegrees` останется от старой, и рамка уедет — это выглядит
     * ровно как двойное применение угла, хотя причина другая.
     */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        анализ?.targetRotation = display.rotation
    }

    override fun onDestroy() {
        super.onDestroy()
        // Закрытие уходит на тот же поток, что и вызовы: иначе `close` может прийтись на
        // середину разбора кадра, а интерпретатор этого не переживёт.
        исполнитель.execute { зрение?.close() }
        зрение = null
        исполнитель.shutdown()
    }

    private companion object {
        const val ТЕГ = "CarLensDemo"
        const val ПЕРИОД_ЭМБЕДДИНГА_МС = 500L
    }
}
