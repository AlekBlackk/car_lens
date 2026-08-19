// Плагины объявлены в корне без применения: версия задаётся один раз на всё дерево,
// а модули подключают их через `alias(...)` из `gradle/libs.versions.toml`.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}
