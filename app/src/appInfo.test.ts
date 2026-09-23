import { describe, expect, it } from 'vitest'
import { версияПриложения } from './appInfo'

describe('версияПриложения', () => {
  it('имя и код версии из нативного модуля', () => {
    expect(версияПриложения({ versionName: '0.4', versionCode: 4 })).toBe('0.4 (4)')
  })

  it('без модуля — явная пометка, а не пустая строка', () => {
    expect(версияПриложения(undefined)).toBe('неизвестна: нет модуля CarlensAppInfo')
  })
})
