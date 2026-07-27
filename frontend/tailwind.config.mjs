export default {
  content: ["./src/**/*.{astro,html,js,jsx,ts,tsx,md,mdx}"],
  theme: {
    extend: {
      colors: {
        bg: "#100e0c",
        panel: "#17140f",
        line: "#2a251d",
        fg: "#ece6da",
        muted: "#8a8378",
        sparse: "#e8a33d",
        dense: "#3ecf8e",
        hybrid: "#b48ce0",
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
