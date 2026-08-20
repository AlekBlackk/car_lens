plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

// Хеши моделей вшиваются на этапе сборки: они отвечают на вопрос «те ли модели попали
// в APK, что раскладывались», а не на вопрос про tampering — эталон лежит в том же
// артефакте, что и проверяемый файл (§10 дизайна).
val манифестМоделей = rootProject.file("models/models_manifest.json")
val ключиМоделей = listOf("classifier", "detector", "index", "meta")
val сообщениеОбОтсутствии =
    "нет models/models_manifest.json — разложи модели:\n" +
        "uv run python -m scripts.stage_assets --run <каталог прогона> " +
        "--detector <detector_fp32.tflite> --out app/android/models"

android {
    namespace = "com.carlens.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.carlens.demo"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"

        // Поля объявляются в обоих случаях, а отсутствие манифеста роняет не конфигурацию,
        // а задачу `checkModelsManifest` ниже: Gradle конфигурирует все подпроекты перед
        // любой задачей, поэтому отказ отсюда уронил бы и `:vision:testDebugUnitTest`, и
        // джобу CI, которым модели не нужны.
        val поляМоделей: Map<String, Pair<String, String>> =
            if (манифестМоделей.exists()) {
                val манифест = groovy.json.JsonSlurper().parse(манифестМоделей) as Map<*, *>
                val файлы = манифест["files"] as Map<*, *>
                ключиМоделей.associateWith { ключ ->
                    val запись = файлы[ключ] as Map<*, *>
                    // Касты строгие: пропущенный ключ обязан уронить сборку здесь, а не
                    // протечь в APK литералом «null» вместо эталонной суммы.
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
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    sourceSets { getByName("main").assets.srcDir(rootProject.file("models")) }
}

// Отказ на фазе выполнения, а не конфигурации: проверка висит в зависимостях `preBuild`,
// поэтому падает любая сборка `:demo` и не падает ничего в `:vision`. Собирать APK без
// моделей нечего — ассетов нет, а пустые поля `BuildConfig` не с чем сверять.
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
    implementation(project(":vision"))
    implementation(libs.androidx.activity)
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)
}
