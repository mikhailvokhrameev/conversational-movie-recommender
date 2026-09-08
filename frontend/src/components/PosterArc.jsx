export default function PosterArc({ posters }) {
  const fanned = posters.filter((_, i) => i % 2 === 0)
  const step = 360 / fanned.length

  return (
    <div
      className="poster-arc shrink-0"
      aria-hidden="true"
      style={{
        '--arc-h': 'clamp(300px, 38vw, 460px)',
        '--arc-r': 'clamp(340px, 40vw, 560px)',
        '--card-w': 'clamp(160px, 18vw, 240px)',
        '--card-h': 'clamp(240px, 27vw, 360px)',
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
