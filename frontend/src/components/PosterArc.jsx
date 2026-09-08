export default function PosterArc({ posters }) {
  const step = 360 / posters.length

  return (
    <div
      className="poster-arc"
      aria-hidden="true"
      style={{
        '--arc-h': 'clamp(160px, 22vw, 260px)',
        '--arc-r': 'clamp(300px, 85vw, 1000px)',
        '--card-w': 'clamp(84px, 9vw, 132px)',
        '--card-h': 'clamp(126px, 13.5vw, 198px)',
        '--arc-duration': '200s',
      }}
    >
      <div className="poster-orbit-wheel">
        {posters.map((src, i) => (
          <div
            key={src}
            className="poster-card rounded-2xl overflow-hidden shadow-lg shadow-black/20 ring-1 ring-black/10"
            style={{ '--angle': `${i * step}deg` }}
          >
            <img src={src} alt="" className="w-full h-full object-cover" loading="lazy" />
          </div>
        ))}
      </div>
    </div>
  )
}
