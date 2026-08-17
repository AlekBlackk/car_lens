/**
 * Печать заголовка и первых классов индекса эталонов читалкой из `dist/`.
 *
 * Существует для приёмки M-1.3: собранный Python файл обязан читаться в
 * JavaScript-рантайме. Отдельный файл, а не `node -e`, потому что читалка —
 * ESM-модуль, а хеш-функция передаётся в `loadIndex` параметром.
 */

import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'

import { loadIndex } from '../dist/vision/catalogIndex.js'

const [dir] = process.argv.slice(2)
if (!dir) {
  console.error('использование: node app/scripts/dump-index.mjs <каталог с catalog_index.bin>')
  process.exit(2)
}

const sha256 = (data) => createHash('sha256').update(data).digest('hex')
const index = loadIndex(
  new Uint8Array(readFileSync(`${dir}/catalog_index.bin`)),
  new Uint8Array(readFileSync(`${dir}/catalog_meta.json`)),
  sha256,
)
console.log(index.header)
console.log(index.classes.slice(0, 5).map((c) => c.display))
