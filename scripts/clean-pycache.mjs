/**
 * scripts/clean-pycache.mjs — remove Python bytecode before packing.
 *
 * `.npmignore` cannot exclude files inside a directory named in package.json's
 * `files` array, so `src/__pycache__/*.pyc` shipped in the tarball (measured: two
 * .pyc files, 88 kB of a 89 kB package). This runs as `prepack`, so any `npm pack`
 * or `npm publish` produces a clean artefact without anyone remembering to delete
 * them by hand.
 */
import { readdirSync, rmSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = fileURLToPath(new URL('..', import.meta.url))
let removed = 0

function walk(dir) {
  let entries
  try {
    entries = readdirSync(dir, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    if (entry.name === 'node_modules' || entry.name === '.git') continue
    const full = join(dir, entry.name)
    if (entry.isDirectory()) {
      if (entry.name === '__pycache__') {
        rmSync(full, { recursive: true, force: true })
        removed++
        continue
      }
      walk(full)
    } else if (entry.name.endsWith('.pyc') || entry.name.endsWith('.pyo')) {
      rmSync(full, { force: true })
      removed++
    }
  }
}

walk(ROOT)
console.log(`clean-pycache: removed ${removed} item(s)`)
