/* Ballwatcher Screener service worker: makes the page installable and usable with a weak connection.
   - The page itself: network first (so updates show up right away), cached copy if offline.
   - Data (JSON/CSV from GitHub): network first, cached copy if offline, so the last boards still open.
   - Icons, fonts: cache first. */
const VERSION = "screener-2026-10-23c";
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./icons/icon-192.png", "./icons/icon-512.png", "./icons/apple-touch-icon.png"];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
const networkFirst = async req => {
  const cache = await caches.open(VERSION);
  try {
    const res = await fetch(req);
    if (res && res.ok) cache.put(stripBust(req), res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(stripBust(req)) || await cache.match(stripBust(req), { ignoreSearch: true });
    if (hit) return hit;
    throw err;
  }
};
const cacheFirst = async req => {
  const cache = await caches.open(VERSION);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res && (res.ok || res.type === "opaque")) cache.put(req, res.clone());
  return res;
};
// the page adds ?t=<time> to data requests to dodge stale caches; store them under one key
function stripBust(req) {
  const u = new URL(req.url); u.searchParams.delete("t"); u.searchParams.delete("v"); u.hash = "";
  return u.toString();
}
self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const u = new URL(req.url);
  if (u.pathname.includes("/admin")) return;                      // never touch the site's admin
  if (req.mode === "navigate") { e.respondWith(networkFirst(req).catch(() => caches.match("./index.html"))); return; }
  if (u.hostname === "raw.githubusercontent.com" || /\.(json|csv)$/.test(u.pathname)) { e.respondWith(networkFirst(req)); return; }
  if (/fonts\.(googleapis|gstatic)\.com$/.test(u.hostname) || /\/icons\//.test(u.pathname) || u.hostname.endsWith("espncdn.com")) { e.respondWith(cacheFirst(req)); return; }
});
