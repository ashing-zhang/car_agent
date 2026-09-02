// Vite 配置 - 开发时将 /api 代理到后端 FastAPI (端口 8000)
// 运行指南:
//   安装依赖: cd frontend && npm install
//   开发模式: npm run dev   (http://localhost:5173)
//   生产构建: npm run build (输出到 frontend/dist)
//   预览构建: npm run preview

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/metrics': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
