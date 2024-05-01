import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react'
import dotenv from 'dotenv';

export default defineConfig({
  plugins: [react()],
  define: {
    'process.env': dotenv.config().parsed
  }
});

