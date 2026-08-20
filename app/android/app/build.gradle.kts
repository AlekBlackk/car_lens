// Минимальный RN-модуль: собирает JS-бандл (`../..` — корень `app/`, где лежит
// `package.json`) и React Native-рантайм в APK. Значения по умолчанию (root, entryFile,
// hermesEnabled/newArchEnabled из `gradle.properties`) совпадают с раскладкой этого
// репозитория, менять их незачем.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    id("com.facebook.react")
}

// `app/package.json` уже объявляет react-native-vision-camera и react-native-worklets-core
// обычными зависимостями (они понадобятся задачам 3-11), поэтому settings-автолинковка
// (`autolinkLibrariesFromCommand()`) генерирует `PackageList.java` с их пакетами
// безусловно, вне зависимости от того, вызван код камеры уже или нет. Без
// `autolinkLibrariesWithApp()` эти нативные модули не попадают в classpath `:app`, и
// сгенерированный `PackageList.java` не компилируется — импортируемых классов просто нет.
react {
    autolinkLibrariesWithApp()
}

android {
    namespace = "com.carlens.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.carlens.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // Будущие задачи (3-11) добавят сюда вызовы FrameProcessorPlugin из `:vision`.
    implementation(project(":vision"))
    implementation("com.facebook.react:react-android")
    implementation("com.facebook.react:hermes-android")
}
