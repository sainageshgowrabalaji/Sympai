import type { NextConfig } from "next";

// "export" builds plain HTML, CSS and JavaScript into out/. There is no server
// code on Vercel at all. The page runs in the browser and talks straight to
// the SympAI server on your laptop.
const config: NextConfig = {
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true },
};

export default config;
