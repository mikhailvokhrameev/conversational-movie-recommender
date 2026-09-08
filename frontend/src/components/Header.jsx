export default function Header({ onHomeClick, themeSlot }) {
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-surface">
      <button
        onClick={onHomeClick}
        className="font-display text-base sm:text-lg text-ink hover:text-amber transition-colors duration-150
                   focus-visible:outline-2 focus-visible:outline-amber focus-visible:outline-offset-2"
      >
        cmr
      </button>
      {themeSlot}
    </header>
  )
}
