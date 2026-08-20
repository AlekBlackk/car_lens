package com.carlens.app

import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.fabricEnabled
import com.facebook.react.defaults.DefaultReactActivityDelegate

class MainActivity : ReactActivity() {

  // Имя корневого компонента, зарегистрированного в JS через `AppRegistry.registerComponent`
  // (см. `index.js` и `app.json` в корне `app/`).
  override fun getMainComponentName(): String = "CarLensApp"

  // `DefaultReactActivityDelegate` включает New Architecture (Fabric) одним флагом
  // `fabricEnabled`, значение которого приходит из `newArchEnabled` в `gradle.properties`.
  override fun createReactActivityDelegate(): ReactActivityDelegate =
      DefaultReactActivityDelegate(this, mainComponentName, fabricEnabled)
}
