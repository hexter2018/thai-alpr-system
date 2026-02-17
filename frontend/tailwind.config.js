export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary:{
          50: '#eff6ff',
          300: '#93c5fd',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
        },
        success: {
          600: '#16a34a',
          700: '#15803d',
        },
        danger: {
          600: '#dc2626',
          700: '#b91c1c',
        },
      },
    },
  },
  plugins: [],
};