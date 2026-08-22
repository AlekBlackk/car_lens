package com.carlens.app.storage

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

/** Пакет ради одного модуля: локальный код автолинковка не видит, регистрируем руками. */
class CarlensStoragePackage : ReactPackage {

    override fun createNativeModules(контекст: ReactApplicationContext): List<NativeModule> =
        listOf(CarlensStorageModule(контекст))

    override fun createViewManagers(
        контекст: ReactApplicationContext
    ): List<ViewManager<*, *>> = emptyList()
}
