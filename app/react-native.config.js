/**
 * Конфигурация React Native CLI. Явно фиксируем расположение Android-проекта и имя
 * Gradle-модуля приложения: `android/` в этом репозитории — не типовой одиночный
 * RN-проект, рядом с `:app` там же лежат `:vision` и `:demo` (M-1.5), и autolinking
 * не должен пытаться трактовать их как часть RN-приложения.
 */
module.exports = {
  project: {
    android: {
      sourceDir: './android',
      appName: 'app',
    },
  },
}
