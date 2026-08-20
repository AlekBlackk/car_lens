// Плагины объявлены в корне без применения: версия задаётся один раз на всё дерево,
// а модули подключают их через `alias(...)` из `gradle/libs.versions.toml`.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}

// RNGP регистрирует на корневом проекте задачи автолинковки и бандлинга JS. Версию
// плагина задаёт сам RNGP (через composite-сборку из `settings.gradle.kts`), поэтому
// зависимость идёт через buildscript-classpath без версии, а не через
// `gradle/libs.versions.toml`, как остальные плагины дерева.
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.facebook.react:react-native-gradle-plugin")
    }
}

apply(plugin = "com.facebook.react.rootproject")

// Сторонние Android-модули из `node_modules` (react-native-vision-camera,
// react-native-worklets-core, ...) резолвят версии SDK легаси-паттерном
// `rootProject.ext.<name>` — так делает вся RN-экосистема, а не только шаблон RN
// 0.76.5, наши модули (`:vision`, `:demo`, `:app`) на это не полагаются и берут версии
// из `gradle/libs.versions.toml`. Без этого блока `react-native-worklets-core` собирает
// свой CMake-код (`hermestooling`) под API 21 по умолчанию NDK, а prefab-библиотека
// `hermestooling` требует minSdk >= 24 — сборка падает уже на конфигурации CMake.
extra["minSdkVersion"] = 26
extra["compileSdkVersion"] = 35
extra["targetSdkVersion"] = 35
extra["kotlinVersion"] = libs.versions.kotlin.get()
