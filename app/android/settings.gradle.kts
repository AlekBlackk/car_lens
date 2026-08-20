pluginManagement {
    // Composite-сборка RNGP: settings-плагин автолинковки (`com.facebook.react.settings`)
    // и root-плагин (`com.facebook.react.rootproject`) резолвятся отсюда, а не с Maven —
    // так делает и сам шаблон RN 0.76.5.
    includeBuild("../node_modules/@react-native/gradle-plugin")
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("com.facebook.react.settings")
}
// Осторожно: типобезопасный `configure<T> { ... }` ждёт лямбду-с-receiver'ом, а не лямбду
// с параметром — `{ ex -> ex.autolinkLibrariesFromCommand() }` не компилируется (задача 1,
// заметки спайка §3). Автолинковка собирает пакеты вроде react-native-vision-camera и
// react-native-worklets-core в Gradle-дерево из `app/package.json`.
extensions.configure<com.facebook.react.ReactSettingsExtension> { autolinkLibrariesFromCommand() }

dependencyResolutionManagement {
    // Было FAIL_ON_PROJECT_REPOS до этой задачи: `:vision`/`:demo` сами репозиториев не
    // объявляют, и запрет молчал. Плагин `com.facebook.react` модуля `:app` добавляет
    // свой maven-репозиторий на уровне проекта (локальные артефакты react-android/
    // hermes-android) — под FAIL_ON_PROJECT_REPOS конфигурация падает с "repository
    // 'maven' was added by plugin 'com.facebook.react'". PREFER_SETTINGS разрешает
    // такие репозитории, оставляя приоритет за перечисленными здесь.
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "carlens-android"
include(":app", ":vision", ":demo")
// Второй, «настоящий» includeBuild (не в pluginManagement): он подставляет RNGP как
// проект вместо координат `com.facebook.react:react-native-gradle-plugin` в classpath
// корневого `build.gradle.kts` — без него `apply(plugin = "com.facebook.react.rootproject")`
// там не резолвится.
includeBuild("../node_modules/@react-native/gradle-plugin")
