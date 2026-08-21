package com.carlens.vision

import android.content.Context
import android.os.SystemClock
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.SharedArray
import com.mrousavy.camera.frameprocessors.VisionCameraProxy

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
 * `frame.orientation.уголВыпрямления()` — та же функция, что использует `CarVisionDetectPlugin`
 * (объявлена там `internal`, поэтому доступна и здесь без импорта: оба файла в одном
 * пакете `com.carlens.vision`). Второй копии этого маппинга здесь нет намеренно.
 *
 * ## Число совпадений на бокс (находка I1 финального ревью M-1.6)
 *
 * `CatalogIndex.knn` отдаёт `min(topK, numClasses)` совпадений, а не ровно `topK`
 * (комментарий на месте, `CatalogIndex.kt`). Буфер `matches` раньше выделялся размером
 * `count * topK * 2`, но заполнялся по фактическому (меньшему на маленьком каталоге)
 * числу — хвост буфера оставался нулями, TS отдавал их как настоящие совпадения
 * `{classIdx: 0, sim: 0}`, а при двух боксах совпадения второго сдвигались в область
 * первого. `CarVisionMatchPlugin` эту же семантику уже решал правильно (отдавал
 * фактическое `count`), здесь исправлено симметрично: фактическое число совпадений на
 * бокс уезжает наружу отдельным полем `matchCount`, и буфер укладывается по нему, а не
 * по `topK`. Оно одинаково для каждого бокса одного вызова — один индекс, один `topK`,
 * — поэтому одного числа на весь ответ достаточно; `check` ниже документирует это
 * предположение и упал бы, если `CatalogIndex.knn` вдруг перестанет быть таким.
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
        val уголГравитации = frame.orientation.уголВыпрямления(уголУстройстваИзПараметров(params))
        val уголЭкрана = frame.orientation.уголВыпрямления(0)

        val боксы = (0 until плоские.size / 5).take(МАКСИМУМ_КРОПОВ).map { номер ->
            val сдвиг = номер * 5
            Box(
                плоские[сдвиг], плоские[сдвиг + 1], плоские[сдвиг + 2],
                плоские[сдвиг + 3], плоские[сдвиг + 4],
            )
        }

        return CarVisionHolder.подЗамком(context) { зрение ->
            // Боксы приехали из JS в координатах экрана — там их и рисовали. Кроп же
            // берётся из кадра, выпрямленного по гравитации, значит их надо вернуть в ту
            // систему, из которой детектор их и получил. Без этого шага кроп уезжает
            // мимо машины ровно на угол поворота телефона, причём тихо: подписи просто
            // становятся хуже.
            val кадрЭкрана = Rotation.upright(frame.image.width, frame.image.height, уголЭкрана)
            val кГравитации = Rotation.betweenUpright(уголЭкрана, уголГравитации)
            val боксыКропа = боксы.map { Rotation.rotateBox(it, кадрЭкрана, кГравитации) }
            // Обратного поворота здесь нет по той же причине, что прямого нет в плагине
            // детекции: боксы пришли в координатах вывода, а `embedAndMatch` с тем же
            // `уголКадра` работает в той же системе координат. Любой доворот тут увёл бы
            // кроп мимо машины, причём тихо — подписи стали бы просто хуже.
            val результаты = зрение.embedAndMatch(frame.image, уголГравитации, боксыКропа, pad, topK)
            val фактическийTopK = результаты.firstOrNull()?.matches?.size ?: 0
            val (совпадения, числаСовпадений) =
                floatSharedArray(proxy, результаты.size * фактическийTopK * 2)
            val поля = HashMap<String, Any>()
            результаты.forEachIndexed { номер, результат ->
                val слот = слоты[номер] ?: SharedArray(proxy, результат.embedding.size * 4)
                    .also { слоты[номер] = it }
                слот.asFloatBuffer().put(результат.embedding)
                поля["emb$номер"] = слот
                check(результат.matches.size == фактическийTopK) {
                    "матчей у бокса $номер: ${результат.matches.size} против $фактическийTopK " +
                        "у бокса 0 — один вызов embedAndMatch обязан отдавать одинаковое число " +
                        "совпадений на каждый бокс (один индекс, один topK)"
                }
                результат.matches.forEach { совпадение ->
                    числаСовпадений.put(совпадение.classIdx.toFloat())
                    числаСовпадений.put(совпадение.sim)
                }
            }
            поля["matches"] = совпадения
            поля["matchCount"] = фактическийTopK
            ответМоста(SystemClock.elapsedRealtimeNanos() - начало, результаты.size, поля)
        }
    }

    private companion object {
        const val МАКСИМУМ_КРОПОВ = 2
    }
}
