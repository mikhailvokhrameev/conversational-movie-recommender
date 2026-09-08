import { useEffect, useState } from 'react'
import MovieCard from './MovieCard'
import { LOADING_MESSAGES } from '../data/loadingMessages'

const LOADING_MESSAGE_INTERVAL_MS = 6767

function pickNextLoadingMessage(current) {
  if (LOADING_MESSAGES.length <= 1) return LOADING_MESSAGES[0]
  let next
  do {
    next = LOADING_MESSAGES[Math.floor(Math.random() * LOADING_MESSAGES.length)]
  } while (next === current)
  return next
}

function RetrievalSkeleton() {
  const [loadingMessage, setLoadingMessage] = useState(() => pickNextLoadingMessage())

  useEffect(() => {
    const interval = setInterval(() => {
      setLoadingMessage(current => pickNextLoadingMessage(current))
    }, LOADING_MESSAGE_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [])

  return (
    <p
      key={loadingMessage}
      className="text-sm text-muted"
      style={{ animation: `loading-message-fade ${LOADING_MESSAGE_INTERVAL_MS}ms ease-in-out` }}
    >
      {loadingMessage}
    </p>
  )
}

export default function ChatMessage({ message, isStreaming, isLast }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] sm:max-w-md px-4 py-2.5 rounded-2xl rounded-br-sm bg-surface text-ink text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    )
  }

  const hasMovies = message.movies?.length > 0
  const showCaret = isStreaming && isLast && !message.error
  const showSkeleton = !message.explanation && showCaret && !hasMovies

  return (
    <div className="space-y-4 max-w-2xl">
      {hasMovies && (
        <div className="relative -mx-4">
          <div className="flex gap-3 overflow-x-auto py-1 px-4 snap-x snap-mandatory scrollbar-thin">
            {message.movies.map(movie => (
              <div key={movie.id} className="snap-start">
                <MovieCard movie={movie} />
              </div>
            ))}
            <div className="flex-shrink-0 w-1" aria-hidden="true" />
          </div>
          <div className="pointer-events-none absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-bg to-transparent" />
        </div>
      )}

      {message.explanation && (
        <div className="text-sm leading-relaxed text-ink/90 whitespace-pre-wrap">
          {message.explanation}
          {showCaret && (
            <span
              className="inline-block w-0.5 h-4 bg-amber ml-0.5 align-text-bottom"
              style={{ animation: 'pulse-caret 1s ease-in-out infinite' }}
            />
          )}
        </div>
      )}

      {showSkeleton && <RetrievalSkeleton />}

      {message.error && (
        <p className="text-sm text-error">
          {message.error}
        </p>
      )}
    </div>
  )
}
