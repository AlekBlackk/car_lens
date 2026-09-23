/**
 * Версия APK для экрана «Настройки». Номер живёт в `build.gradle.kts` и приходит константами
 * нативного модуля `CarlensAppInfo` (`AppInfoModule.kt`), а не копией в JS.
 *
 * Модуль без `react-native`: чтение `NativeModules` остаётся на экране, а здесь — только
 * форматирование, которое проверяется тестом без RN-рантайма.
 */
export type КонстантыВерсии = { versionName: string; versionCode: number }

export function версияПриложения(константы: КонстантыВерсии | undefined): string {
  // Пометка вместо пустоты: незарегистрированный модуль иначе выглядел бы как «версии нет».
  if (константы == null) return 'неизвестна: нет модуля CarlensAppInfo'
  return `${константы.versionName} (${константы.versionCode})`
}
