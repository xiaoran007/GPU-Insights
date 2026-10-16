import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss()],
  base: mode === 'workbench' ? './' : '/GPU-Insights/',
  publicDir: mode === 'workbench' ? false : 'public',
  build: {
    outDir: mode === 'workbench' ? 'dist-workbench' : '../docs',
    emptyOutDir: true,
    ...(mode === 'workbench' ? {
      rollupOptions: { input: fileURLToPath(new URL('./workbench.html', import.meta.url)) },
    } : {}),
  },
}))
