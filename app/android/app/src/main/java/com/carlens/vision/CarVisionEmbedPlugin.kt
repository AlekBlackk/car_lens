package com.carlens.vision

import android.content.Context
import android.os.SystemClock
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.SharedArray
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import java.nio.ByteOrder

/**
 * Эмбеддинг и top-K для боксов, выбранных троттлингом в JS.
 *
 * Слоты выделяются один раз и переиспользуются: аллокация килобайта на каждый кроп
 * каждого кадра повторила бы болезнь препроцессинга M-1.5. Отсюда контракт, который
 * обязан знать JS: **эмбеддинг слота действителен до следующего вызова**, и перед
 * укладкой в состояние трека его надо скопировать.
 *
 * Слотов два, а не один: при одном оба результата двухбоксового вызова смотрели бы в
 * одну память. Тот же класс ошибки на Kotlin-уровне ловит тест
 * `два_бокса_дают_разные_эмбеддинги` из M-1.5.
 *
 * `frame.orientation.toDegrees()` — та же функция, что использует `CarVisionDetectPlugin`
 * (объявлена там `internal`, поэтому доступна и здесь без импорта: оба файла в одном
 * пакете `com.carlens.vision`). Второй копии этого маппинга здесь нет намеренно.
 */
class CarVisionEmbedPlugin(
    private val proxy: VisionCameraProxy,
    private val context: Context,
) : FrameProcessorPlugin() {

    private val слоты = arrayOfNulls<SharedArray>(МАКСИМУМ_КРОПОВ)

    override fun callback(frame: Frame, params: Map<String, Any>?): Any {
        val начало = SystemClock.elapsedRealtimeNanos()
        require(!frame.isMirrored) { "фронтальная камера не поддерживается" }
        val плоские = (params?.get("boxes") as? List<*>)?.map { (it as Number).toFloat() }
            ?: emptyList()
        require(плоские.size % 5 == 0) { "боксы: ${плоские.size} чисел, не кратно 5" }
        val pad = (params?.get("pad") as? Number)?.toFloat() ?: 0.08f
        val topK = (params?.get("topK") as? Number)?.toInt() ?: 5
        val уголКадра = frame.orientation.toDegrees()

        val боксы = (0 until плоские.size / 5).take(МАКСИМУМ_КРОПОВ).map { номер ->
            val сдвиг = номер * 5
            Box(
                плоские[сдвиг], плоские[сдвиг + 1], плоские[сдвиг + 2],
                плоские[сдвиг + 3], плоские[сдвиг + 4],
            )
        }

        return CarVisionHolder.подЗамком(context) { зрение ->
            // Обратного поворота здесь нет по той же причине, что прямого нет в плагине
            // детекции: боксы пришли в координатах вывода, а `embedAndMatch` с тем же
            // `уголКадра` работает в той же системе координат. Любой доворот тут увёл бы
            // кроп мимо машины, причём тихо — подписи стали бы просто хуже.
            val результаты = зрение.embedAndMatch(frame.image, уголКадра, боксы, pad, topK)
            val ответ = HashMap<String, Any>()
            val совпадения = SharedArray(proxy, результаты.size * topK * 2 * 4)
            val числаСовпадений =
                совпадения.byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
            результаты.forEachIndexed { номер, результат ->
                val слот = слоты[номер] ?: SharedArray(proxy, результат.embedding.size * 4)
                    .also { слоты[номер] = it }
                слот.byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
                    .put(результат.embedding)
                ответ["emb$номер"] = слот
                результат.matches.forEach { совпадение ->
                    числаСовпадений.put(совпадение.classIdx.toFloat())
                    числаСовпадений.put(совпадение.sim)
                }
            }
            ответ["nativeNs"] = (SystemClock.elapsedRealtimeNanos() - начало).toDouble()
            ответ["count"] = результаты.size
            ответ["matches"] = совпадения
            ответ
        }
    }

    private companion object {
        const val МАКСИМУМ_КРОПОВ = 2
    }
}
