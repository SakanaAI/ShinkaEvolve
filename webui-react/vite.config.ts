import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/list_databases': 'http://localhost:8000',
      '/get_programs': 'http://localhost:8000',
      '/get_meta_files': 'http://localhost:8000',
      '/get_meta_content': 'http://localhost:8000',
      '/download_meta_pdf': 'http://localhost:8000',
    },
  },
})
