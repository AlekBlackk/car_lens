plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.carlens.vision"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    sourceSets {
        // Модели лежат в одном каталоге на всё дерево и подключаются туда, где нужны:
        // инструментальному APK библиотеки — как свои ассеты. Отсутствие каталога не
        // ошибка: JVM-тестам и сборке библиотеки модели не нужны.
        getByName("androidTest").assets.srcDir(rootProject.file("models"))
        // Фикстура формата индекса общая с TypeScript-читалкой: две реализации одного
        // формата обязаны проверяться на одном файле, иначе они разъедутся.
        getByName("test").resources.srcDir(rootProject.file("../../tests/fixtures/index"))
        // Эталон паритета (задача 10) едет в инструментальный APK ассетами: тест на
        // устройстве читает те же PNG и тот же `golden.json`, что собрал питон.
        getByName("androidTest").assets.srcDir(rootProject.file("../../tests/fixtures/device"))
    }
}

dependencies {
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.litert.gpu.api)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.androidx.annotation)
    testImplementation(libs.junit)
    androidTestImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.junit)
    androidTestImplementation(libs.androidx.test.runner)
}
