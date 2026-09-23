package com.carlens.app

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.uimanager.ViewManager

/**
 * Версия приложения для экрана «Настройки». Константы, а не метод: значения известны на
 * сборке и не меняются за запуск, RN отдаёт их JS синхронно при создании модуля.
 *
 * Источник один — `versionName`/`versionCode` в `build.gradle.kts`; копия номера в JS
 * разошлась бы с APK при первом же подъёме версии.
 */
class AppInfoModule(контекст: ReactApplicationContext) : ReactContextBaseJavaModule(контекст) {

    override fun getName(): String = "CarlensAppInfo"

    override fun getConstants(): Map<String, Any> = mapOf(
        "versionName" to BuildConfig.VERSION_NAME,
        "versionCode" to BuildConfig.VERSION_CODE,
    )
}

/** Пакет ради одного модуля: локальный код автолинковка не видит, регистрируем руками. */
class AppInfoPackage : ReactPackage {

    override fun createNativeModules(контекст: ReactApplicationContext): List<NativeModule> =
        listOf(AppInfoModule(контекст))

    override fun createViewManagers(
        контекст: ReactApplicationContext
    ): List<ViewManager<*, *>> = emptyList()
}
