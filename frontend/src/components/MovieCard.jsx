export default function MovieCard({ movie }) {
  const year = movie.release_date?.slice(0, 4)

  return (
    <a
      href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex-shrink-0 w-44 sm:w-48 rounded-lg bg-surface overflow-hidden
                 flex flex-col
                 transition-colors duration-200 hover:bg-elevated
                 focus-visible:outline-2 focus-visible:outline-amber focus-visible:outline-offset-2"
    >
      <div className="aspect-[3/4] bg-elevated flex items-center justify-center overflow-hidden">
        {movie.poster_url ? (
          <img
            src={movie.poster_url}
            alt={movie.serial_name}
            loading="lazy"
            className="w-full h-full object-cover"
          />
        ) : (
          <svg className="w-8 h-8 text-muted opacity-30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="2" width="20" height="20" rx="2" />
            <path d="M7 2v20M17 2v20M2 7h20M2 12h20M2 17h20" />
          </svg>
        )}
      </div>

      <div className="p-2.5 flex flex-col gap-1.5 flex-1">
        <h3 className="font-display text-sm leading-tight text-ink group-hover:text-amber transition-colors duration-200">
          {movie.serial_name}
        </h3>

        <div className="flex items-center gap-1.5 text-[0.7rem] text-muted">
          {year && <span>{year}</span>}
          {movie.vote_average > 0 && (
            <>
              <span className="opacity-40">·</span>
              <span>★ {movie.vote_average.toFixed(1)}</span>
            </>
          )}
        </div>

        <div className="flex flex-wrap gap-1">
          {movie.genres?.slice(0, 2).map(genre => (
            <span
              key={genre}
              className="px-1.5 py-0.5 text-[0.6rem] font-medium tracking-wide
                         bg-amber-muted text-amber rounded"
            >
              {genre}
            </span>
          ))}
        </div>

        <div className="flex items-center justify-between mt-auto">
          <span className="text-[0.6rem] font-medium text-amber tracking-wide">
            {(movie.score * 100).toFixed(0)}% match
          </span>
        </div>
      </div>
    </a>
  )
}
