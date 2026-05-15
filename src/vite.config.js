import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // CRITICAL: without base './' all asset URLs are absolute (/assets/...)
  // which breaks when Electron loads index.html via file:// in the packaged app.
  base: './',
})
