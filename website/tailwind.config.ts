import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        asu: {
          maroon: "#8c1d40",
          gold: "#ffc627",
          ink: "#1f2937",
          mist: "#f8f5f2",
          line: "#e7dfd7",
        },
      },
      boxShadow: {
        panel: "0 18px 40px -28px rgba(69, 31, 20, 0.28)",
      },
      fontFamily: {
        sans: ["Avenir Next", "Segoe UI", "Helvetica Neue", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
