import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const SWIPE_THRESHOLD_PX = 40

/** Wrap-normalised distance from the active slot, so the deck never runs out of flanking cards. */
function slotOffset(index, active, count) {
  const half = Math.floor(count / 2)
  let offset = index - active
  if (offset > half) offset -= count
  if (offset < -half) offset += count
  return offset
}

function slotClass(offset) {
  if (offset === 0) return 'deck-card--active'
  if (offset === -1) return 'deck-card--prev'
  if (offset === 1) return 'deck-card--next'
  return 'deck-card--parked'
}

function PosterFace({ movie }) {
  if (!movie.poster_url) {
    return (
      <span className="w-full h-full flex items-center justify-center bg-elevated">
        <svg
          className="w-8 h-8 text-muted opacity-30"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-hidden="true"
        >
          <rect x="2" y="2" width="20" height="20" rx="2" />
          <path d="M7 2v20M17 2v20M2 7h20M2 12h20M2 17h20" />
        </svg>
      </span>
    )
  }
  return (
    <img
      src={movie.poster_url}
      alt=""
      loading="lazy"
      draggable="false"
      className="w-full h-full object-cover select-none"
    />
  )
}

const CARD_SURFACE =
  'block w-full aspect-[2/3] rounded-2xl overflow-hidden bg-elevated ' +
  'shadow-lg shadow-black/20 ring-1 ring-black/10 ' +
  'focus-visible:outline-2 focus-visible:outline-amber focus-visible:outline-offset-2'

function Chevron({ direction }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="w-8 h-8 sm:w-11 sm:h-11"
      fill="none"
      stroke="currentColor"
      strokeWidth="3.25"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={direction === 'prev' ? 'M15.5 4.5 8 12l7.5 7.5' : 'M8.5 4.5 16 12l-7.5 7.5'} />
    </svg>
  )
}

const ARROW_CLASS =
  'deck-arrow absolute top-1/2 -translate-y-1/2 z-10 p-1 text-ink ' +
  'transition-[color,transform] duration-150 hover:text-amber active:scale-90 ' +
  'disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:text-ink disabled:active:scale-100 ' +
  'focus-visible:outline-2 focus-visible:outline-amber focus-visible:outline-offset-2 rounded-md'

export default function MovieCarousel({ movies, descriptions, isStreaming, onActiveChange }) {
  const [active, setActive] = useState(0)
  const [announcement, setAnnouncement] = useState('')
  const pointerRef = useRef(null)
  const swipedRef = useRef(false)
  const announcedRef = useRef(false)
  const onActiveChangeRef = useRef(onActiveChange)
  onActiveChangeRef.current = onActiveChange

  const count = movies?.length ?? 0
  const safeIndex = count > 0 ? Math.min(active, count - 1) : 0

  const go = useCallback(
    (delta) => {
      if (count < 2) return
      setActive((current) => {
        const from = Math.min(current, count - 1)
        return (from + delta + count) % count
      })
    },
    [count],
  )

  const activeMovie = count > 0 ? movies[safeIndex] : null
  const activeTitle = activeMovie?.serial_name ?? ''

  /* Reports the card in view (also on mount, i.e. index 0 of a new list) so the chat can say what "this" is. */
  useEffect(() => {
    if (activeMovie) onActiveChangeRef.current?.(activeMovie)
  }, [activeMovie])

  useEffect(() => {
    if (!announcedRef.current) {
      announcedRef.current = true
      return
    }
    if (!activeTitle) return
    setAnnouncement(`Movie ${safeIndex + 1} of ${count}: ${activeTitle}`)
  }, [safeIndex, count, activeTitle])

  function handleKeyDown(e) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      go(-1)
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      go(1)
    }
  }

  function handlePointerDown(e) {
    if (e.pointerType === 'mouse' && e.button !== 0) return
    pointerRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY }
    swipedRef.current = false
  }

  function handlePointerUp(e) {
    const start = pointerRef.current
    pointerRef.current = null
    if (!start || start.id !== e.pointerId) return
    const dx = e.clientX - start.x
    const dy = e.clientY - start.y
    if (Math.abs(dx) < SWIPE_THRESHOLD_PX || Math.abs(dx) <= Math.abs(dy)) return
    swipedRef.current = true
    go(dx < 0 ? 1 : -1)
  }

  function handlePointerCancel() {
    pointerRef.current = null
  }

  /* A swipe that ends on the centre poster must not follow its link. */
  function handleClickCapture(e) {
    if (!swipedRef.current) return
    swipedRef.current = false
    e.preventDefault()
    e.stopPropagation()
  }

  if (!count || !activeMovie) return null

  const year = activeMovie.release_date?.slice(0, 4)
  const meta = [
    year,
    activeMovie.vote_average > 0 ? `★ ${activeMovie.vote_average.toFixed(1)}` : null,
    typeof activeMovie.score === 'number' ? `${(activeMovie.score * 100).toFixed(0)}% match` : null,
  ].filter(Boolean)

  const blurb = (descriptions?.[safeIndex] ?? '').trim()
  const catalogDescription = (activeMovie.description ?? '').trim()

  let body
  if (blurb) {
    body = (
      <div className="prose prose-sm prose-chat max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{blurb}</ReactMarkdown>
      </div>
    )
  } else if (isStreaming) {
    body = (
      <p className="text-sm text-muted">
        Writing about <span className="italic">{activeTitle}</span>
        <span
          className="inline-block w-0.5 h-3.5 bg-amber ml-1 align-text-bottom"
          style={{ animation: 'pulse-caret 1s ease-in-out infinite' }}
        />
      </p>
    )
  } else if (catalogDescription) {
    body = (
      <div className="prose prose-sm prose-chat max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{catalogDescription}</ReactMarkdown>
      </div>
    )
  } else {
    body = <p className="text-sm text-muted">No description available for this title.</p>
  }

  return (
    <section
      className="flex flex-col items-center gap-5 sm:gap-7"
      aria-roledescription="carousel"
      aria-label="Recommended movies"
      onKeyDown={handleKeyDown}
    >
      <div className="relative w-full pt-4 pb-1 sm:pt-6">
        <button
          type="button"
          onClick={() => go(-1)}
          disabled={count < 2}
          aria-label="Previous movie"
          className={`${ARROW_CLASS} left-0`}
        >
          <Chevron direction="prev" />
        </button>

        <div
          className="deck-stage mx-auto rounded-2xl focus-visible:outline-2 focus-visible:outline-amber focus-visible:outline-offset-8"
          tabIndex={0}
          role="group"
          aria-label={`Movie ${safeIndex + 1} of ${count}: ${activeTitle}`}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          onClickCapture={handleClickCapture}
        >
          {movies.map((movie, index) => {
            const offset = slotOffset(index, safeIndex, count)
            const isActive = offset === 0
            const className = `deck-card ${slotClass(offset)}`

            if (isActive) {
              return (
                <div key={movie.id} className={className}>
                  <a
                    href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    title={`${movie.serial_name} on TMDB`}
                    className={CARD_SURFACE}
                  >
                    <PosterFace movie={movie} />
                  </a>
                </div>
              )
            }

            return (
              <div key={movie.id} className={className} aria-hidden="true">
                <button
                  type="button"
                  tabIndex={-1}
                  onClick={() => setActive(index)}
                  className={`${CARD_SURFACE} cursor-pointer`}
                >
                  <PosterFace movie={movie} />
                </button>
              </div>
            )
          })}
        </div>

        <button
          type="button"
          onClick={() => go(1)}
          disabled={count < 2}
          aria-label="Next movie"
          className={`${ARROW_CLASS} right-0`}
        >
          <Chevron direction="next" />
        </button>
      </div>

      {count > 1 && (
        <div className="flex items-center gap-1">
          {movies.map((movie, index) => (
            <button
              key={movie.id}
              type="button"
              onClick={() => setActive(index)}
              aria-label={`Show movie ${index + 1}: ${movie.serial_name}`}
              aria-current={index === safeIndex ? 'true' : undefined}
              className="p-1.5 rounded-full focus-visible:outline-2 focus-visible:outline-amber focus-visible:outline-offset-1"
            >
              <span
                className={`block w-1.5 h-1.5 rounded-full bg-ink transition-opacity duration-200 ${
                  index === safeIndex ? 'opacity-100' : 'opacity-25 hover:opacity-60'
                }`}
              />
            </button>
          ))}
        </div>
      )}

      <div className="w-full rounded-2xl border border-ink/15 px-5 py-5 sm:px-8 sm:py-6 min-h-[13rem] text-center">
        <h3 className="font-display text-xl sm:text-2xl leading-tight text-ink">{activeTitle}</h3>

        {meta.length > 0 && (
          <p className="mt-1.5 text-xs text-muted">{meta.join(' · ')}</p>
        )}

        {activeMovie.genres?.length > 0 && (
          <ul className="mt-2.5 flex flex-wrap justify-center gap-1 list-none p-0">
            {activeMovie.genres.slice(0, 4).map((genre) => (
              <li
                key={genre}
                className="px-1.5 py-0.5 text-[0.6rem] font-medium tracking-wide bg-amber-muted text-amber rounded"
              >
                {genre}
              </li>
            ))}
          </ul>
        )}

        <div className="mt-3.5 text-sm leading-relaxed text-ink/90">{body}</div>
      </div>

      <span className="sr-only" role="status" aria-live="polite">
        {announcement}
      </span>
    </section>
  )
}
