import java.security.KeyStore
import java.security.MessageDigest

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
val ключиМоделей = listOf("classifier", "detector", "index", "meta", "thresholds", "calibration")
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

    // Отпечаток сертификата, которым подписана сборка, вшивается в BuildConfig и
    // сверяется на старте (`SigningIdentity.kt`).
    //
    // Чем это НЕ является: защитой от подмены моделей. Перепаковщик правит и модель, и
    // это поле, подписывает своим ключом — сверка сойдётся. Приложение не может быть
    // корнем доверия для самого себя, для настоящего анти-tampering нужен внешний якорь
    // (аттестация, Play Integrity), а его в MVP нет и не планируется: §9 спеки — в релизе
    // сети нет вовсе.
    //
    // Чем является: контролем сборочной конфигурации. Ловит рассинхрон между тем, чем
    // сборка подписывалась, и тем, что стоит на устройстве.
    //
    // Пустая строка означает «проверка выключена», и причина у неё не одна: keystore
    // отсутствует ИЛИ не читается. Отладочного ключа на машине может не быть, и ронять
    // из-за этого конфигурацию Gradle нельзя — но по самому пустому значению причину не
    // отличить, поэтому каждая ветка ниже перед выходом пишет свой `logger.warn`. Молча
    // проверка не выключается никогда.
    //
    // Отдельно про хрупкость: `KeyStore.getInstance("JKS")` читает PKCS12-хранилище — а
    // современный `debug.keystore` бывает именно такой — только через режим совместимости
    // хранилищ в JDK. С `keystore.type.compat=false` загрузка упадёт на валидном keystore,
    // отпечаток выйдет пустым, и проверка выключит себя сама.
    //
    // Значение читается из signingConfig'а `debug`, а `buildConfigField` кладёт его во ВСЕ
    // варианты. Верно это лишь пока `buildTypes.release` подписан отладочным ключом (см.
    // ниже). Появится настоящий релизный ключ — каждый релиз получит отпечаток, который
    // никогда не совпадёт, и `check(...)` в `SigningIdentity` начнёт ронять запуск у
    // пользователей; тогда отпечаток надо считать из signingConfig'а текущего варианта.
    val отпечатокПодписи: String =
        runCatching {
            val подпись = signingConfigs.getByName("debug")
            val файл = подпись.storeFile ?: run {
                logger.warn("carlens: у signingConfig 'debug' не задан storeFile — сверка подписи выключена")
                return@runCatching ""
            }
            if (!файл.exists()) {
                logger.warn("carlens: keystore '$файл' не существует — сверка подписи выключена")
                return@runCatching ""
            }
            val хранилище = KeyStore.getInstance("JKS")
            файл.inputStream().use { хранилище.load(it, подпись.storePassword?.toCharArray()) }
            val алиас = подпись.keyAlias ?: хранилище.aliases().nextElement()
            val сертификат = хранилище.getCertificate(алиас) ?: run {
                logger.warn("carlens: в keystore '$файл' нет сертификата под алиасом '$алиас' — сверка подписи выключена")
                return@runCatching ""
            }
            MessageDigest.getInstance("SHA-256")
                .digest(сертификат.encoded)
                .joinToString("") { байт -> "%02x".format(байт) }
        }.getOrElse { причина ->
            // Сюда попадают только брошенные исключения: чтение хранилища (неверный
            // storePassword, битый или не-JKS файл), отсутствие алиасов, кодирование
            // сертификата. Печатается настоящая причина, а не догадка про «нет keystore».
            logger.warn(
                "carlens: keystore не читается (${причина::class.simpleName}: ${причина.message})" +
                    " — сверка подписи выключена"
            )
            ""
        }
    defaultConfig.buildConfigField("String", "SIGNING_CERT_SHA256", "\"$отпечатокПодписи\"")

    // Релизный вариант подписан отладочным ключом намеренно. Это не путь в магазин, а
    // единственный способ снять с телефона осмысленные числа и вообще пользоваться
    // приложением руками: отладочный APK несёт `android:debuggable=true`, ART держит
    // процесс в отладочном режиме и режет JIT, отчего детектор на этом же телефоне даёт
    // 237 мс на кадр против 47-92 мс в релизе (bench/latency.md, M-1.5). Тот же приём и
    // по той же причине уже применён в `:vision` через `carlens.testRelease`.
    buildTypes {
        getByName("release") {
            signingConfig = signingConfigs.getByName("debug")
            isMinifyEnabled = false
        }
    }
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
