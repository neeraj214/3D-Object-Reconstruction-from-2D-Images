/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          primary: '#3b82f6', // Bright blue
          secondary: '#6366f1', // Indigo
          accent: '#06b6d4', // Cyan neon
          dark: '#0f172a', // Slate 900
          darker: '#020617', // Slate 950
        },
        surface: {
          DEFAULT: '#1e293b', // Slate 800
          soft: '#334155', // Slate 700
          glass: 'rgba(15, 23, 42, 0.7)',
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['Outfit', 'Inter', 'sans-serif']
      },
      borderRadius: {
        xl: '1rem',
        '2xl': '1.5rem',
        '3xl': '2rem'
      },
      boxShadow: {
        soft: '0 4px 20px -2px rgba(0, 0, 0, 0.2)',
        glow: '0 0 20px rgba(59, 130, 246, 0.5)',
        'glow-accent': '0 0 20px rgba(6, 182, 212, 0.5)'
      },
      animation: {
        'blob': 'blob 7s infinite',
        'fade-in': 'fadeIn 0.5s ease-out forwards',
      },
      keyframes: {
        blob: {
          '0%': { transform: 'translate(0px, 0px) scale(1)' },
          '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
          '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
          '100%': { transform: 'translate(0px, 0px) scale(1)' },
        },
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        }
      }
    },
  },
  plugins: [],
}
