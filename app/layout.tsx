import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AegisIRT',
  description: 'Autonomous AI for Clinical Trial UI Testing',
  generator: 'v0.dev',
  icons: {
    icon: [
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon.ico', type: 'image/x-icon' },
    ],
    apple: '/apple-touch-icon.png',
    shortcut: '/favicon.ico',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/site.webmanifest" />
        <meta name="theme-color" content="#ffffff" />
        {/* Optional: More meta tags can go here */}
      </head>
      <body>{children}</body>
    </html>
  );
}
