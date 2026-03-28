/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        wine: {
          DEFAULT: '#722F37',
          dark: '#5A252C',
          light: '#8B3A42',
          50: '#FDF2F3',
          100: '#F9E3E5',
        },
        gold: {
          DEFAULT: '#C9A96E',
          light: '#D4BF8A',
          dark: '#A68B4B',
        },
        cream: {
          DEFAULT: '#FDF8F0',
          dark: '#F0E6D3',
        },
        charcoal: '#2D2926',
        sage: {
          DEFAULT: '#8B9D77',
          light: '#A8B896',
        },
      },
      fontFamily: {
        serif: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
