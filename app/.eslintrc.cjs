// Конфигурация ESLint v8 (legacy-формат, не flat config — пакет `eslint` пинован
// на ^8.57.1, см. app/package.json).
module.exports = {
  root: true,
  env: {
    es2021: true,
    node: true,
  },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2021,
    sourceType: 'module',
  },
  plugins: ['@typescript-eslint'],
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended'],
  rules: {},
  overrides: [
    {
      // Global Constraint плана M-1.6: никаких `any` в app/src/vision/.
      files: ['src/vision/**/*.ts', 'src/vision/**/*.tsx'],
      rules: {
        '@typescript-eslint/no-explicit-any': 'error',
      },
    },
  ],
  ignorePatterns: ['node_modules/', 'android/', '*.cjs'],
}
