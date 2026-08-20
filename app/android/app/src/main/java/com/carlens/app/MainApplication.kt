package com.carlens.app

import android.app.Application
import com.carlens.vision.CarVisionDetectPlugin
import com.facebook.react.PackageList
import com.facebook.react.ReactApplication
import com.facebook.react.ReactHost
import com.facebook.react.ReactNativeHost
import com.facebook.react.ReactPackage
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.load
import com.facebook.react.defaults.DefaultReactHost.getDefaultReactHost
import com.facebook.react.defaults.DefaultReactNativeHost
import com.facebook.react.soloader.OpenSourceMergedSoMapping
import com.facebook.soloader.SoLoader
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

class MainApplication : Application(), ReactApplication {

  override val reactNativeHost: ReactNativeHost =
      object : DefaultReactNativeHost(this) {
        override fun getPackages(): List<ReactPackage> =
            PackageList(this).packages.apply {
              // Пакеты, которые автолинковка не подхватывает сама, добавляются здесь
              // вручную, например: add(MyReactNativePackage())
            }

        override fun getJSMainModuleName(): String = "index"

        override fun getUseDeveloperSupport(): Boolean = BuildConfig.DEBUG

        override val isNewArchEnabled: Boolean = BuildConfig.IS_NEW_ARCHITECTURE_ENABLED
        override val isHermesEnabled: Boolean = BuildConfig.IS_HERMES_ENABLED
      }

  override val reactHost: ReactHost
    get() = getDefaultReactHost(applicationContext, reactNativeHost)

  override fun onCreate() {
    super.onCreate()
    SoLoader.init(this, OpenSourceMergedSoMapping)
    if (BuildConfig.IS_NEW_ARCHITECTURE_ENABLED) {
      // При включённой New Architecture нужно дополнительно загрузить нативную точку входа.
      load()
    }
    // Регистрация до создания любого Frame Processor'а: JS ищет плагин по имени
    // "carvisionDetect" через `VisionCameraProxy.initFrameProcessorPlugin`, и до этой
    // регистрации поиск вернёт `undefined` (см. `app/src/vision/plugins.ts`).
    FrameProcessorPluginRegistry.addFrameProcessorPlugin("carvisionDetect") { proxy, _ ->
      CarVisionDetectPlugin(proxy, applicationContext)
    }
  }
}
