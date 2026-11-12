import type { NextConfig } from "next";

if (process.env.VERCEL === "1") {
  const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!apiUrl || !apiUrl.startsWith("https://") || /localhost|127\.0\.0\.1/.test(apiUrl)) {
    throw new Error("Set NEXT_PUBLIC_API_BASE_URL to the HTTPS backend URL before deploying to Vercel.");
  }
}

const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
