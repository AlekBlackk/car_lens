// Минимальный RN-модуль: собирает JS-бандл (`../..` — корень `app/`, где лежит
// `package.json`) и React Native-рантайм в APK. Значения по умолчанию (root, entryFile,
// hermesEnabled/newArchEnabled из `gradle.properties`) совпадают с раскладкой этого
// репозитория, менять их незачем.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    id("com.facebook.react")
}

// Хеши моделей вшиваются на этапе сборки: они отвечают на вопрос «те ли модели попали
// в APK, что раскладывались», а не на вопрос про tampering — эталон лежит в том же
// артефакте, что и проверяемый файл. Тот же механизм, что в `:demo`.
val манифестМоделей = rootProject.file("models/models_manifest.json")
val ключиМоделей = listOf("classifier", "detector", "index", "meta")
val сообщениеОбОтсутствии =
    "нет models/models_manifest.json — разложи модели:\n" +
        "uv run python -m scripts.stage_assets --run <каталог прогона> " +
        "--detector <detector_fp32.tflite> --out app/android/models"

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
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Поля объявляются в обоих случаях, а отсутствие манифеста роняет не конфигурацию,
        // а задачу `checkModelsManifest` ниже: Gradle конфигурирует все подпроекты перед
        // любой задачей, поэтому отказ отсюда уронил бы и `:vision:testDebugUnitTest`.
        val поляМоделей: Map<String, Pair<String, String>> =
            if (манифестМоделей.exists()) {
                val манифест = groovy.json.JsonSlurper().parse(манифестМоделей) as Map<*, *>
                val файлы = манифест["files"] as Map<*, *>
                ключиМоделей.associateWith { ключ ->
                    val запись = файлы[ключ] as Map<*, *>
                    (запись["sha256"] as String) to (запись["name"] as String)
                }
            } else {
                ключиМоделей.associateWith { "" to "" }
            }
        поляМоделей.forEach { (ключ, значения) ->
            val (sha256, имя) = значения
            buildConfigField("String", "SHA_${ключ.uppercase()}", "\"$sha256\"")
            buildConfigField("String", "NAME_${ключ.uppercase()}", "\"$имя\"")
        }
    }
    buildFeatures { buildConfig = true }
    sourceSets { getByName("main").assets.srcDir(rootProject.file("models")) }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

// Отказ на фазе выполнения, а не конфигурации: проверка висит в зависимостях задачи
// упаковки ассетов, поэтому падает только сборка APK и не падает ничего в `:vision`.
val проверкаМанифеста =
    tasks.register("checkModelsManifest") {
        group = "verification"
        description = "Проверяет, что модели разложены в app/android/models/."
        val манифест = манифестМоделей
        val сообщение = сообщениеОбОтсутствии
        doLast {
            if (!манифест.exists()) {
                throw GradleException(сообщение)
            }
        }
    }
// Только упаковка APK требует моделей: компиляция Kotlin их не читает, а CI собирает
// именно её, без разложенных ассетов.
tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }
    .configureEach { dependsOn(проверкаМанифеста) }

dependencies {
    // Будущие задачи (3-11) добавят сюда вызовы FrameProcessorPlugin из `:vision`.
    implementation(project(":vision"))
    implementation("com.facebook.react:react-android")
    implementation("com.facebook.react:hermes-android")

    // Задача 7: `SharedArrayContractTest` использует `androidx.camera.core.ImageProxy`/
    // `ImageInfo` (тестовый дубль поверх `android.media.Image`) и JUnit-раннер
    // инструментальных тестов. `:vision` зависимости на `camera-core` не объявляет вовсе —
    // источник её в classpath `:app` это `implementation "androidx.camera:camera-core:..."`
    // самой react-native-vision-camera (`node_modules/react-native-vision-camera/android/
    // build.gradle:187`), автолинкованной через `autolinkLibrariesWithApp()`. `implementation`
    // не течёт транзитивно в androidTest-конфигурацию `:app` (проверено компиляцией:
    // `Unresolved reference: ImageProxy`), поэтому нужна явная androidTest-зависимость,
    // просто чтобы дать компилятору сам факт зависимости на классы `ImageProxy`/`ImageInfo`.
    //
    // Версию она НЕ контролирует: react-native-vision-camera форсирует свой собственный
    // `camera-core` строгим (`strictly`) констрейнтом androidx.camera release train до
    // `1.5.0-alpha03`, и он побеждает пин `libs.camera.core` = 1.4.1 из `libs.versions.toml`
    // (`camerax`) — реально резолвится 1.5.0-alpha03 (проверено `:app:dependencies
    // --configuration debugAndroidTestCompileClasspath`). Интерфейсы `ImageProxy`/`ImageInfo`
    // в 1.5.0-alpha03 идентичны использованным в тесте (плюс один необязательный
    // default-метод `toBitmap()`), так что расхождение версий безвредно — не форсировать
    // 1.4.1 здесь: это сломало бы саму react-native-vision-camera, которой нужна именно
    // 1.5.0-alpha03.
    androidTestImplementation(libs.camera.core)
    androidTestImplementation(libs.androidx.test.junit)
    androidTestImplementation(libs.androidx.test.runner)
}
