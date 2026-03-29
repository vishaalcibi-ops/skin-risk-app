/**
 * Skin Risk AI — Service Worker
 * Enables offline caching and PWA install prompt.
 */
const CACHE_NAME = 'skinrisk-v1.2';
const STATIC_ASSETS = [
    '/',
    '/dashboard',
    '/history',
    '/pharmacy',
    '/academy',
    '/profile',
    '/faq',
    '/support',
    '/technology',
    '/static/style.css',
    '/static/manifest.json',
    'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
];

// Install event: cache static assets
self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('[SW] Pre-caching static assets');
            return cache.addAll(STATIC_ASSETS).catch((err) => {
                console.warn('[SW] Some cache items failed to load:', err);
            });
        })
    );
});

// Activate event: clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
            );
        })
    );
    self.clients.claim();
});

// Fetch event: serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;

    // Skip non-GET requests (e.g., POST for /predict)
    if (request.method !== 'GET') return;

    // Network-first for API routes
    const apiRoutes = ['/predict', '/chat/send', '/logout', '/login', '/signup'];
    if (apiRoutes.some(route => request.url.includes(route))) {
        return; // Pass through to network
    }

    event.respondWith(
        caches.match(request).then((cached) => {
            if (cached) return cached;

            return fetch(request).then((response) => {
                // Cache successful static asset responses
                if (response.ok && request.url.includes('/static/')) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
                }
                return response;
            }).catch(() => {
                // If offline and no cache, show a friendly offline page header
                if (request.headers.get('accept').includes('text/html')) {
                    return new Response(
                        '<html><body style="background:#0a0f1e;color:white;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column"><h2>📡 You are offline</h2><p>Please check your internet connection and try again.</p></body></html>',
                        { headers: { 'Content-Type': 'text/html' } }
                    );
                }
            });
        })
    );
});
