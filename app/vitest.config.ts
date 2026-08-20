import { defineConfig } from 'vitest/config'

// Минимальная конфигурация: находит и запускает *.test.ts в src/.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    exclude: ['node_modules/**'],
  },
})
