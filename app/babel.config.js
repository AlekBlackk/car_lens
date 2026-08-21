module.exports = {
  presets: ['module:@react-native/babel-preset'],
  // Без этого плагина директива 'worklet' не транслируется в объект воркета:
  // VisionCamera получает функцию с пустым `__initData.code` и падает на
  // `Compiling JS failed: 1:1:invalid empty parentheses '( )'` при первом кадре
  // (проверено на устройстве, M-1.6 приёмка). Плагин обязателен по README
  // react-native-worklets-core.
  plugins: [['react-native-worklets-core/plugin']],
}
