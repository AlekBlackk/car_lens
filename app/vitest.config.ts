import { defineConfig } from 'vitest/config'

// Минимальная конфигурация: находит и запускает *.test.ts в src/. Миграция
// `catalogIndex.test.ts` (написан под node:test, а не vitest) — задача 11, не эта.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    exclude: ['src/vision/catalogIndex.test.ts', 'node_modules/**'],
  },
})
