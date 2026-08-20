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

// Отказ на фазе выполнения, а не конфигурации: проверка висит в зависимостях `preBuild`,
// поэтому падает любая сборка `:app` и не падает ничего в `:vision`.
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
tasks.named("preBuild") { dependsOn(проверкаМанифеста) }

dependencies {
    // Будущие задачи (3-11) добавят сюда вызовы FrameProcessorPlugin из `:vision`.
    implementation(project(":vision"))
    implementation("com.facebook.react:react-android")
    implementation("com.facebook.react:hermes-android")
}
