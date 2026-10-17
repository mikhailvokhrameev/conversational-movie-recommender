# Frontend Reference (`frontend/`)

React 19 + Vite + Tailwind CSS 4 chat interface for the movie recommender.
Communicates with the backend via SSE streaming (`POST /api/chat/`).
Dark cinematic theme with OKLCH color tokens.

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| React | 19 | UI components |
| Vite | 8 | Build tool + dev server |
| Tailwind CSS | 4 | Utility-first styling via `@tailwindcss/vite` |
| Google Fonts | - | DM Serif Display (display) + Inter (body) |

No additional UI libraries, state management, or routing. Single-page app.

## Project Structure

```
frontend/
├── index.html              HTML entry point (lang="en")
├── vite.config.js          Vite config with Tailwind plugin + API proxy
├── Dockerfile              Node 20 Alpine, runs dev server on port 3000
├── src/
│   ├── main.jsx            React root mount
│   ├── index.css           Tailwind imports, OKLCH tokens, keyframes
│   ├── App.jsx             Top-level layout: header, welcome screen or chat + input
│   ├── data/
│   │   ├── posters.js      Static poster image list for the welcome-screen hero
│   │   └── loadingMessages.js  LOADING_MESSAGES flavor-text pool for the retrieval skeleton
│   ├── hooks/
│   │   ├── useChat.js      SSE streaming hook (fetch + ReadableStream)
│   │   └── useTheme.js     Dark/light theme state (localStorage-backed)
│   └── components/
│       ├── Header.jsx         Home button + theme toggle slot
│       ├── ThemeToggle.jsx    Dark/light theme switch button
│       ├── ChatInput.jsx      Textarea + send button form
│       ├── ChatMessage.jsx    User bubble or assistant response (intro text + carousel)
│       ├── MovieCarousel.jsx  Fanned poster deck, arrows, dots, description panel
│       ├── WelcomeScreen.jsx  Empty state: arc-text heading, input, poster arc hero
│       └── PosterArc.jsx      Decorative rotating arc of static poster images (welcome screen only)
└── public/
    └── favicon.svg
```

## Design Tokens

Defined in `src/index.css` via Tailwind's `@theme` directive. All colors use OKLCH.

| Token | Value | Role |
|-------|-------|------|
| `--color-bg` | `oklch(0.08 0 0)` | Page background (near-black, zero chroma) |
| `--color-surface` | `oklch(0.15 0 0)` | Cards, bubbles, input fields |
| `--color-elevated` | `oklch(0.20 0 0)` | Hover states, poster placeholder |
| `--color-amber` | `oklch(0.75 0.16 55)` | Primary accent (CTAs, score badges, active states) |
| `--color-amber-hover` | `oklch(0.80 0.18 55)` | Send button hover |
| `--color-amber-muted` | `oklch(0.75 0.16 55 / 0.15)` | Genre chip background |
| `--color-ink` | `oklch(0.95 0 0)` | Primary text |
| `--color-muted` | `oklch(0.65 0 0)` | Secondary text, metadata |
| `--color-error` | `oklch(0.65 0.2 25)` | Error messages |

Fonts:
- `--font-display`: DM Serif Display (movie titles, headings)
- `--font-body`: Inter (chat messages, UI text)

## Components

### `App.jsx`

Top-level layout. Full-viewport flex column (`h-svh`). Renders `Header`
(home button + `ThemeToggle` slot) above everything else, then shows
`WelcomeScreen` when no messages exist, otherwise a scrollable message list.
Input bar is sticky at the bottom with `backdrop-blur-sm` and semi-transparent
background.

Auto-scrolls to bottom on new messages via `useEffect` + `scrollRef`.

### `Header.jsx` / `ThemeToggle.jsx`

`Header` is a thin bar: a home button (resets the chat via `onHomeClick`) and
a slot for `ThemeToggle`, a sun/moon icon button that flips `useTheme`'s
`dark`/`light` state (persisted to `localStorage`).

### `useChat.js` (hook)

Manages chat state and SSE streaming. Returns
`{ messages, isStreaming, sendMessage, sessionId, reset }`. `reset` aborts any
in-flight stream and clears `messages`/`sessionId`/the session token, returning
to the empty state; `App.jsx` wires it to `Header`'s home button.

**Message shape:**
```js
// User message
{ id: number, role: 'user', content: string }

// Assistant message
{
  id: number, role: 'assistant',
  movies: Movie[],          // from the `movies` event
  explanation: string,      // intro line, accumulated from `token` events
  descriptions: string[],   // per-movie text, accumulated from `blurb` events (index-aligned with `movies`)
  intent: object,
  error: string|null
}
```

**SSE protocol:** Uses `fetch` + `ReadableStream` (not `EventSource`, because
`EventSource` only supports GET). Parses the SSE text protocol manually:
`event: <type>\ndata: <json>\n\n`. Event types:
- `session`: sets `session_id` and stashes `session_token` (sent back as the
  `X-Session-Token` header on the next request)
- `movies`: sets session_id, movies array, and intent; also reads/sets
  `session_token` from the payload (same as `session`)
- `token`: appends text to the explanation string (the intro line)
- `intro_final`: replaces the explanation with `text` -- sent only when the reply had no
  `[[n]]` markers, so the intro would otherwise hold every description; the cards then
  show catalog copy
- `blurb`: appends `text` to `descriptions[index]`, the per-movie description
- `done`: ignored (stream closes naturally)
- `error`: sets error message on the assistant message

Plain conversational turns (`general_chat` / `follow_up`) receive only `token` events,
so `movies` stays empty and no carousel renders -- just markdown prose.

A "nothing suitable" turn (the backend's `no_match: true`: the relevance gate or the hard
SQL filters left no results) arrives as a `movies` event with `movies: []` plus one `token` with a fixed
message. The client needs no special case and does not read `no_match`: an empty
`movies` array renders no carousel, and the token text shows as an ordinary assistant
reply (which also clears the loading skeleton).

Each message gets a stable numeric `id` (module-level counter) for React keys.

### `ChatMessage.jsx`

Renders a single message. Two layouts:

**User messages:** Right-aligned bubble with `bg-surface`, rounded corners
(larger radius top, small bottom-right for chat bubble shape).

**Assistant messages:** Left-aligned, up to `max-w-2xl`. Prose comes *first*, then the
carousel -- the reverse of the older cards-then-text order. Contains:
1. The intro line: streamed `token` text, markdown-rendered, with an amber pulse caret
   during streaming. One or two sentences that name no titles.
2. `MovieCarousel`, once the `movies` event has arrived
3. `RetrievalSkeleton`: a single rotating loading-message string (from
   `data/loadingMessages.js`'s `LOADING_MESSAGES`) that cross-fades to a new
   phrase every few seconds, shown while waiting for first data
4. Error message in `text-error` if stream failed

A turn with no movies (plain chat) renders step 1 alone.

### `MovieCarousel.jsx`

The whole recommendation block: a fanned poster deck above a description panel.
Replaces the former horizontal `MovieCard` shelf (`MovieCard.jsx` is gone; the poster
card now lives inside this file and the metadata it used to carry moved into the panel).

**Deck geometry.** Cards are poster-only. The active movie sits centered at full scale
and full opacity; the previous and next movies peek out from behind it, scaled down,
rotated slightly outward and semi-transparent -- a fanned hand of cards, the still
counterpart to `PosterArc`'s revolving wheel. Each poster renders `movie.poster_url`
when present and falls back to a film-reel SVG icon when it is null.

**Navigation.** Left/right chevron arrows flank the deck and step the active index; the
same step is bound to the arrow keys when the deck has focus, and to a horizontal swipe
on touch. The index wraps in both directions, so the deck never dead-ends at either
edge. A row of position dots under the deck shows where you are in the five results.

**Description panel.** Sits under the deck and describes the *centered* movie only:
- Title in serif (`font-display`), linking to the movie's TMDB page
  (`themoviedb.org/movie/{tmdb_id}`, new tab)
- Metadata: year + TMDB rating (`★ {vote_average}`), shown only when `vote_average > 0`
- Score badge: `{score * 100}% match` in amber
- Genre chips, amber-muted background
- The description body, resolved through this fallback chain:
  1. the streamed `blurb` text for that index, if any has arrived;
  2. a "still writing" state while the stream is running and that slot is still empty;
  3. the movie's catalog `description` from the `movies` event -- the path taken when
     the model emitted no `[[n]]` markers at all (see [API](api.md)).

Focus-visible rings in amber on the arrows, dots, and title link.

### `ChatInput.jsx`

Form with textarea + send button. Enter submits (Shift+Enter for newline).
Textarea has hidden `<label>` for screen readers. Send button is 44px
(minimum touch target), amber background, arrow icon. Both elements have
`focus-visible` outlines. Disabled during streaming.

### `WelcomeScreen.jsx`

Empty-state hero. An SVG arc-text heading ("conversational movie
recommender" along a curved `<textPath>`, `sr-only` duplicated as plain text
for accessibility), a pill-shaped input form (rotated -1deg, straightens on
focus) that submits directly via `onSuggestionClick`, three suggestion-chip
buttons below it, and a `PosterArc` decorative hero filling the rest of the
viewport.

### `PosterArc.jsx`

Purely decorative (`aria-hidden`). Arranges the images from `data/posters.js`
in a circular arc via CSS custom properties (`--angle` per card) and rotates
the whole wheel slowly (`--arc-duration: 200s`). Posters are static assets
served from `frontend/public/posters/`, unrelated to the movie catalog data -- see
`PRODUCT.md` for sourcing.

## Dev Server

Vite dev server runs on port 3000 with API proxy:

```js
proxy: { '/api': 'http://backend:8000' }
```

In Docker, the frontend container connects to the backend service.
The `VITE_API_URL` env var is available but the hook currently uses
the relative `/api/chat/` path (proxied by Vite in dev, needs nginx
or similar in production).

## Accessibility

- Hidden `<label>` on chat textarea
- `aria-label` on send button
- `focus-visible` outlines on all interactive elements (amber color)
- 44px minimum touch targets on buttons
- `prefers-reduced-motion: reduce` disables all animations
- Semantic HTML: `<main>`, `<footer>`, `<form>`, `<h1>`, `<h3>`
- The carousel's movie title is an `<a>` to TMDB (keyboard-focusable, screen-reader-announced)
- Carousel arrows are buttons with `aria-label`s and respond to the arrow keys

## Related

- [API Reference](api.md) for backend endpoints and SSE protocol
- [Core ML Pipeline](core.md) for scoring and recommendation logic
- [Data Models](models.md) for Movie and ChatSession schemas
- [Infrastructure](infrastructure.md) for Docker services
