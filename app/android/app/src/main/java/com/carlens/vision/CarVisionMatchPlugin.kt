package com.carlens.vision

import android.content.Context
import android.os.SystemClock
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.SharedArray
import com.mrousavy.camera.frameprocessors.VisionCameraProxy

/**
 * Совпадения для готового вектора.
 *
 * `Frame` не используется, но обязателен: сигнатура `callback` в VisionCamera 4.6.4
 * требует его. Вынести вызов в обход frame processor нельзя по более важной причине —
 * `CarVision` не потокобезопасен, и вызов с JS-потока пришёлся бы на середину разбора
 * кадра на потоке VisionCamera.
 *
 * Вектор приходит как `ArrayBuffer`, а на стороне Kotlin оказывается `SharedArray`:
 * `JSIJNIConversion.cpp` (строки 60-64 в 4.6.4) заворачивает JSI-ArrayBuffer в него без
 * копирования. Окно `Float32Array` в эту ветку не попадает — `isArrayBuffer` в JSI истинно
 * только для самого буфера, — поэтому JS обязан присылать `.buffer`, и обёртка в
 * `plugins.ts` это делает.
 *
 * Ответ выделяется заново на каждый вызов, а не берётся из переиспользуемого слота, как
 * эмбеддинги в `CarVisionEmbedPlugin`: там слот спасал от килобайта на кроп на каждом
 * кадре, здесь же речь о `count * 2 * 4` = 40 байтах при topK = 5. Заодно снимается
 * контракт времени жизни — возвращённый буфер никто не перезапишет.
 */
class CarVisionMatchPlugin(
    private val proxy: VisionCameraProxy,
    private val context: Context,
) : FrameProcessorPlugin() {

    @Suppress("UNUSED_PARAMETER")
    override fun callback(frame: Frame, params: Map<String, Any>?): Any {
        val начало = SystemClock.elapsedRealtimeNanos()
        val общий = params?.get("embedding") as? SharedArray
            ?: error("embedding: ожидался ArrayBuffer, пришло ${params?.get("embedding")}")
        val topK = (params?.get("topK") as? Number)?.toInt() ?: 5
        val числа = общий.asFloatBuffer()
        val вектор = FloatArray(числа.remaining())
        числа.get(вектор)

        return CarVisionHolder.подЗамком(context) { зрение ->
            // Размерность проверяет сам `matchEmbedding` через `Embeddings.l2NormalizedCopy`,
            // и его текст ошибки точнее любого дубля здесь: он называет и пришедшее число,
            // и размерность индекса.
            val совпадения = зрение.matchEmbedding(вектор, topK)
            // `CatalogIndex.knn` отдаёт `min(topK, numClasses)` совпадений, а не ровно
            // `topK`. Наружу поэтому уезжает фактическое число: проверка длины по `topK`
            // на маленьком каталоге падала бы на совершенно верном ответе.
            val (буфер, плоские) = floatSharedArray(proxy, совпадения.size * 2)
            совпадения.forEach { совпадение ->
                плоские.put(совпадение.classIdx.toFloat())
                плоские.put(совпадение.sim)
            }
            ответМоста(
                SystemClock.elapsedRealtimeNanos() - начало,
                совпадения.size,
                mapOf("matches" to буфер),
            )
        }
    }
}
