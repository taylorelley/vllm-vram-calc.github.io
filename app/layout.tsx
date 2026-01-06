import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'vLLM VRAM Calculator - GPU Memory Planning Tool',
  description: 'Calculate GPU memory requirements for vLLM LLM deployments. Plan VRAM allocation, KV cache capacity, and tensor parallelism configuration.',
  keywords: ['vLLM', 'GPU', 'VRAM', 'LLM', 'memory calculator', 'tensor parallelism', 'KV cache'],
  authors: [{ name: 'vLLM Community' }],
  creator: 'vLLM Community',
  publisher: 'vLLM Community',
  metadataBase: new URL('https://vllm-calc.vercel.app'),
  openGraph: {
    title: 'vLLM VRAM Calculator',
    description: 'Plan GPU memory allocation for LLM deployments',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'vLLM VRAM Calculator',
    description: 'Plan GPU memory allocation for LLM deployments',
  },
  manifest: '/manifest.json',
  icons: {
    icon: [
      { url: '/icons/icon-192x192.png', sizes: '192x192', type: 'image/png' },
      { url: '/icons/icon-512x512.png', sizes: '512x512', type: 'image/png' },
    ],
    apple: [
      { url: '/icons/icon-152x152.png', sizes: '152x152', type: 'image/png' },
    ],
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'vLLM Calc',
  },
};

export const viewport: Viewport = {
  themeColor: '#00ff88',
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        {children}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js').then(
                    function(registration) {
                      console.log('ServiceWorker registration successful');
                    },
                    function(err) {
                      console.log('ServiceWorker registration failed: ', err);
                    }
                  );
                });
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
