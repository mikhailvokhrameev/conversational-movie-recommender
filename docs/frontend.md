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
├── index.html              HTML entry point (lang="ru")
├── vite.config.js          Vite config with Tailwind plugin + API proxy
├── Dockerfile              Node 20 Alpine, runs dev server on port 3000
├── src/
│   ├── main.jsx            React root mount
│   ├── index.css           Tailwind imports, OKLCH tokens, keyframes
│   ├── App.jsx             Top-level layout: header, welcome screen or chat + input
│   ├── data/
│   │   └── posters.js      Static poster image list for the welcome-screen hero
│   ├── hooks/
│   │   ├── useChat.js      SSE streaming hook (fetch + ReadableStream)
│   │   └── useTheme.js     Dark/light theme state (localStorage-backed)
│   └── components/
│       ├── Header.jsx         Home button + theme toggle slot
│       ├── ThemeToggle.jsx    Dark/light theme switch button
│       ├── ChatInput.jsx      Textarea + send button form
│       ├── ChatMessage.jsx    User bubble or assistant response (cards + text)
│       ├── MovieCard.jsx      Poster (or placeholder), title, rating, genres, score, TMDB link
│       ├── WelcomeScreen.jsx  Empty state: arc-text heading, input, suggestion chips, poster arc hero
│       └── PosterArc.jsx      Decorative rotating arc of static poster images (welcome screen only)
└── public/
    ├── favicon.svg
    └── icons.svg
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

Manages chat state and SSE streaming. Returns `{ messages, isStreaming, sendMessage, sessionId }`.

**Message shape:**
```js
// User message
{ id: number, role: 'user', content: string }

// Assistant message
{ id: number, role: 'assistant', movies: Movie[], explanation: string, intent: object, error: string|null }
```

**SSE protocol:** Uses `fetch` + `ReadableStream` (not `EventSource`, because
`EventSource` only supports GET). Parses the SSE text protocol manually:
`event: <type>\ndata: <json>\n\n`. Three event types:
- `movies`: sets session_id, movies array, and intent
- `token`: appends text to explanation string
- `done`: ignored (stream closes naturally)
- `error`: sets error message on the assistant message

Each message gets a stable numeric `id` (module-level counter) for React keys.

### `ChatMessage.jsx`

Renders a single message. Two layouts:

**User messages:** Right-aligned bubble with `bg-surface`, rounded corners
(larger radius top, small bottom-right for chat bubble shape).

**Assistant messages:** Left-aligned, up to `max-w-2xl`. Contains:
1. Horizontal scroll row of `MovieCard` components (with right-edge gradient fade)
2. Streaming explanation text with amber pulse caret during streaming
3. Skeleton loading placeholders (3 pulsing rectangles) when waiting for first data
4. Error message in `text-error` if stream failed

### `MovieCard.jsx`

A link (`<a>`) to the movie's TMDB page (`themoviedb.org/movie/{tmdb_id}`).
Opens in new tab. Structure:
- Poster area: `aspect-[3/4]`, renders the real poster (`movie.poster_url`,
  built server-side from `poster_path`) when present, falls back to a
  film-reel SVG icon when `poster_url` is null
- Title: serif font (`font-display`), 2-line clamp, amber on hover
- Metadata: year + TMDB rating (`★ {vote_average}`), shown only when
  `vote_average > 0`
- Genre chips: up to 2, amber-muted background
- Score badge: `{score * 100}% match` in amber

Fixed width `w-44 sm:w-48`, `flex-shrink-0` for horizontal scrolling.
Focus-visible ring in amber for keyboard navigation.

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
under `frontend/images/`, unrelated to the movie catalog data -- see
`PRODUCT.md` for sourcing.

## Dev Server

Vite dev server runs on port 3000 with API proxy:

```js
proxy: { '/api': 'http://localhost:8000' }
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
- Movie cards are `<a>` elements (keyboard-focusable, screen-reader-announced)

## Related

- [API Reference](api.md) for backend endpoints and SSE protocol
- [Core ML Pipeline](core.md) for scoring and recommendation logic
- [Data Models](models.md) for Movie and ChatSession schemas
- [Infrastructure](infrastructure.md) for Docker services
