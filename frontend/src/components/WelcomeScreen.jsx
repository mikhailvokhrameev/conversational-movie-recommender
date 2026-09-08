import { useState } from 'react'
import PosterArc from './PosterArc'
import { POSTERS } from '../data/posters'

const PLACEHOLDER = 'Something like Interstellar, but a little bit funnier...'

export default function WelcomeScreen({ onSuggestionClick }) {
  const [text, setText] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const value = text.trim()
    if (!value) return
    onSuggestionClick(value)
    setText('')
  }

  return (
    <div className="flex-1 flex flex-col items-center overflow-hidden">
      <div className="flex-1 flex flex-col items-center justify-center gap-6 sm:gap-7 px-4 w-full pb-4">
        <h1 className="sr-only">Conversational movie recommender</h1>

        <svg viewBox="0 0 2000 300" className="relative z-10 w-full max-w-5xl sm:max-w-6xl -mb-8" aria-hidden="true">
          <path id="hero-arc-path" d="M 30 260 Q 1000 30 1970 260" fill="none" />
          <text
            fontFamily="var(--font-display-arc)"
            fontWeight="800"
            fontSize="126"
            fill="var(--color-ink)"
          >
            <textPath
              href="#hero-arc-path"
              startOffset="50%"
              textAnchor="middle"
              textLength="1875"
              lengthAdjust="spacingAndGlyphs"
            >
              conversational movie recommender
            </textPath>
          </text>
        </svg>

        <form onSubmit={handleSubmit} className="w-full max-w-2xl">
          <div
            className="w-full flex items-center gap-3 -rotate-1 focus-within:rotate-0 rounded-full
                       border-2 border-ink bg-bg px-3 py-2 pl-6
                       transition-transform duration-200"
          >
            <label htmlFor="hero-input" className="sr-only">Describe the movie you're looking for</label>
            <input
              id="hero-input"
              type="text"
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder={PLACEHOLDER}
              className="flex-1 min-w-0 bg-transparent text-base sm:text-lg text-ink
                         placeholder:text-muted focus:outline-none"
            />
            <button
              type="submit"
              disabled={!text.trim()}
              className="flex-shrink-0 w-10 h-10 rounded-full bg-ink text-bg
                         flex items-center justify-center
                         hover:bg-amber disabled:opacity-30 disabled:cursor-not-allowed
                         focus-visible:outline-2 focus-visible:outline-amber focus-visible:outline-offset-2
                         transition-colors duration-150"
              aria-label="Send"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </form>
      </div>

      <PosterArc posters={POSTERS} />
    </div>
  )
}
