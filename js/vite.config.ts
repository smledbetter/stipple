import { defineConfig } from "vite";
import { resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      formats: ["es"],
      fileName: () => "index.js",
    },
    outDir: resolve(__dirname, "../src/stipple/_static"),
    emptyOutDir: true,
    target: "esnext",
    minify: false,
    sourcemap: true,
    rollupOptions: {
      output: { inlineDynamicImports: true },
    },
  },
});
