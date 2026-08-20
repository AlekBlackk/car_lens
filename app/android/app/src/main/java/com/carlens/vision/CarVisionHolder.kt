package com.carlens.vision

import android.content.Context
import com.carlens.app.BuildConfig
import java.util.concurrent.locks.ReentrantLock

/**
 * Единственный экземпляр `CarVision` на процесс и замок вокруг него.
 *
 * `CarVision` не потокобезопасен (§5.2 дизайна M-1.5). Все три плагина зовутся с потока
 * VisionCamera, но безопасность обеспечивается замком, а не этой договорённостью:
 * `runAsync` увёл бы вызов на другой поток, и «один поток» перестало бы быть правдой
 * ровно в тот момент, когда кто-то захочет ускорить кадр.
 *
 * Не закрывается никогда: у плагина нет хука жизненного цикла, а `close()` с чужого
 * потока посреди `detect` убивает интерпретатор. Процесс умирает — ресурсы уходят с ним.
 */
internal object CarVisionHolder {

    private val замок = ReentrantLock()

    @Volatile
    private var экземпляр: CarVision? = null

    @Volatile
    private var отказИнициализации: Throwable? = null

    /**
     * Выполнить работу под замком на живом экземпляре.
     *
     * `tryLock`, а не `lock`: занятый замок означает вызов со второго потока, то есть
     * программную ошибку. Блокирующее ожидание спрятало бы её за просевшим fps.
     */
    fun <T> подЗамком(context: Context, работа: (CarVision) -> T): T {
        check(замок.tryLock()) {
            "мост занят другим потоком: для carvision-плагинов запрещён runAsync"
        }
        try {
            отказИнициализации?.let { throw it }
            val живой = экземпляр ?: поднять(context)
            return работа(живой)
        } finally {
            замок.unlock()
        }
    }

    private fun поднять(context: Context): CarVision {
        val ожидаемые = mapOf(
            "classifier" to BuildConfig.SHA_CLASSIFIER,
            "detector" to BuildConfig.SHA_DETECTOR,
            "index" to BuildConfig.SHA_INDEX,
            "meta" to BuildConfig.SHA_META,
        )
        return runCatching { CarVision.load(context.assets, AssetSpec(expected = ожидаемые)) }
            .onSuccess { экземпляр = it }
            .onFailure { отказИнициализации = it }
            .getOrThrow()
    }
}
