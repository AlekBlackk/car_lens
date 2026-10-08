package com.carlens.vision

import com.mrousavy.camera.frameprocessors.SharedArray
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Общая укладка плоского wire-формата моста carvision*.
 *
 * Все три плагина (`CarVisionDetectPlugin`, `CarVisionEmbedPlugin`, `CarVisionMatchPlugin`)
 * кодируют списки чисел одинаково: плоский `SharedArray` в нативном порядке байт плюс
 * обёртка ответа `nativeNs`/`count` (+ свои поля). Вынесено сюда находкой M9 финального
 * ревью ветки M-1.6: пока укладка была продублирована по трём файлам (по пять копий
 * `byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()` и по три копии сборки ответа),
 * задачи 6 и 8 разошлись в понимании одного факта — что `CatalogIndex.knn` отдаёт
 * `min(topK, numClasses)` совпадений, а не ровно `topK` (находка I1, там же). Один файл не
 * гарантирует, что такое расхождение не повторится, но убирает причину, по которой оно
 * возникло тут: одну и ту же операцию просто негде было написать дважды по-разному.
 */

/** Пул `SharedArray` плагина по размеру в байтах: массив на кадр переполнял таблицу ссылок JNI (см. `BufferPool`). */
internal fun sharedArrayPool(proxy: VisionCameraProxy): BufferPool<SharedArray> =
    BufferPool { байт -> SharedArray(proxy, байт) }

/**
 * `SharedArray` на `floatCount` чисел из пула плагина и плоский `FloatBuffer` поверх него, готовый к `put`.
 * Массив тот же, что выдавался на прошлом вызове такого размера: его содержимое действительно
 * до следующего вызова плагина, а все `floatCount` чисел вызывающий обязан записать заново.
 */
internal fun floatSharedArray(пул: BufferPool<SharedArray>, floatCount: Int): Pair<SharedArray, FloatBuffer> {
    val массив = пул.взять(floatCount * 4)
    return массив to массив.asFloatBuffer()
}

/** Плоский `FloatBuffer` поверх уже существующего `SharedArray` — слотов и присланных из JS буферов. */
internal fun SharedArray.asFloatBuffer(): FloatBuffer =
    byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()

/** Каркас ответа моста, общий для всех трёх плагинов: время нативной стороны плюс `count` и свои поля. */
internal fun ответМоста(nativeNs: Long, count: Int, поля: Map<String, Any>): Map<String, Any> {
    val ответ = HashMap<String, Any>(поля.size + 2)
    ответ["nativeNs"] = nativeNs.toDouble()
    ответ["count"] = count
    ответ.putAll(поля)
    return ответ
}
