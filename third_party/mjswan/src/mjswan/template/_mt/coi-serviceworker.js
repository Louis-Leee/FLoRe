/* coi-serviceworker — vendored from https://github.com/gzuidhof/coi-serviceworker (MIT)
 *
 * Only required for GitHub Pages hosting, which cannot set custom response headers.
 * On Netlify / Cloudflare Pages / Vercel the _headers file is sufficient instead.
 * On local `app.launch()` the Python server already sends COOP/COEP — no SW needed.
 */

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (event) => {
  event.waitUntil(
    self.clients.claim().then(() =>
      self.clients.matchAll().then((clients) =>
        clients.forEach((client) => client.postMessage({ type: "activated" }))
      )
    )
  );
});

self.addEventListener("fetch", (event) => {
  // Chrome extensions and non-http(s) requests — skip
  if (!event.request.url.startsWith("http")) return;

  // Avoid a failed fetch for opaque same-origin requests
  if (
    event.request.cache === "only-if-cached" &&
    event.request.mode !== "same-origin"
  ) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Opaque responses (cross-origin, no-cors) — pass through unchanged
        if (response.status === 0) return response;

        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Resource-Policy", "cross-origin");

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      })
      .catch(() => fetch(event.request))
  );
});
