export default function PosterArc({ posters }) {
  const fanned = posters.filter((_, i) => i % 3 === 0)
  const step = 360 / fanned.length

  return (
    <div
      className="poster-arc shrink-0"
      aria-hidden="true"
      style={{
        '--arc-h': 'clamp(390px, 49vw, 598px)',
        '--arc-r': 'clamp(442px, 52vw, 728px)',
        '--card-w': 'clamp(208px, 23vw, 312px)',
        '--card-h': 'clamp(312px, 35vw, 468px)',
        '--arc-duration': '150s',
      }}
    >
      <div className="poster-orbit-wheel">
        {fanned.map((src, i) => (
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
