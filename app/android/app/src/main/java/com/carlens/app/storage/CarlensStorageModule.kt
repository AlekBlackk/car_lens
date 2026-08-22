package com.carlens.app.storage

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.util.concurrent.Executors

/**
 * Единственный нативный модуль приложения: отдаёт JS ключ шифрования локального
 * хранилища.
 *
 * **Почему не frame-плагин.** Три существующих моста (`carvisionDetect` и соседи)
 * регистрируются во `FrameProcessorPluginRegistry` и вызываются из worklet'а с кадром на
 * руках. Ключ нужен один раз при старте и до всякой камеры — это другой род вызова.
 *
 * **Почему обычный `ReactContextBaseJavaModule`, а не TurboModule.** На новой архитектуре
 * RN 0.76 такие модули работают через слой совместимости, а полноценный TurboModule
 * потребовал бы `codegenConfig` и файла спецификации ради одного асинхронного метода.
 *
 * Модуль не подхватывается автолинковкой — он локальный, — и регистрируется явным
 * `add(CarlensStoragePackage())` в `MainApplication.getPackages()`.
 */
class CarlensStorageModule(контекст: ReactApplicationContext) :
    ReactContextBaseJavaModule(контекст) {

    // Один поток, а не пул: вызов ровно один за запуск, и параллелить тут нечего.
    private val исполнитель = Executors.newSingleThreadExecutor()

    override fun getName(): String = ИМЯ

    /**
     * Отдаёт `{ ключ, сброшен }`. Промис, а не синхронный вызов: метод ходит в
     * `AndroidKeyStore` и в `SharedPreferences`, и блокировать на этом JS-поток при старте
     * приложения нельзя.
     */
    @ReactMethod
    fun initStorage(промис: Promise) {
        исполнитель.execute {
            try {
                val итог = StorageKey.получить(reactApplicationContext)
                val ответ =
                    Arguments.createMap().apply {
                        putString("ключ", итог.ключBase64)
                        putBoolean("сброшен", итог.сброшен)
                    }
                промис.resolve(ответ)
            } catch (ошибка: Throwable) {
                // Отказ отдаётся наверх кодом, а не текстом: JS показывает экран отказа и
                // не пытается создать MMKV без ключа.
                промис.reject("storage_key_failed", ошибка)
            }
        }
    }

    override fun invalidate() {
        исполнитель.shutdown()
        super.invalidate()
    }

    companion object {
        const val ИМЯ = "CarlensStorage"
    }
}
