import { createPortal } from 'react-dom';
import { useTheme } from '../../theme/ThemeProvider';

/* ============================================================
   SwagAtmosphere — многослойный фон + декор для режима «свага».
   Портал в <body>, position:fixed, под контентом (z-index:0),
   pointer-events:none — клики не ловит. Рендерится только в swag.

   Ассеты берём из /public/swag через BASE_URL (корректно и при
   деплое в подпапку). Стайлинг слоёв — в swag.css (.sa-*),
   здесь — только пути к картинкам и точечные значения углей/мышей.
   ============================================================ */

const A = `${import.meta.env.BASE_URL}swag/`;

const bg = (file) => ({ backgroundImage: `url(${A}${file})` });

// силуэт летучей мыши (один path из прототипа)
const BAT_PATH =
  'M32 13c2-5 5-8 8-7 1-3 4-5 6-3 0 2-1 3-1 5 4-2 8-2 11 1-4 0-6 2-7 5-3-2-6-1-8 2-2-4-6-4-9 0-3-4-7-4-9 0-2-3-5-4-8-2-1-3 1-5 5-5 0-2 0-3-1-5 2-2 5 0 6 3 3-1 6 2 8 7z';

const bats = [
  { w: 46, h: 26, fill: '#07070a', flap: '.26s', style: { top: '12%', left: '-6%', animation: 'batCross 17s linear 0s infinite' } },
  { w: 30, h: 17, fill: '#08080c', flap: '.32s', style: { top: '22%', left: '-9%', animation: 'batCross 23s linear 5s infinite' } },
  { w: 38, h: 21, fill: '#070709', flap: '.29s', style: { top: '8%', right: '-7%', animation: 'batCross2 27s linear 11s infinite' } },
];

const embers = [
  { bottom: '14%', left: '18%', size: 3, c: '#dfe4ec', glow: 'rgba(210,220,240,.7)', dur: '7s',   delay: '0s'   },
  { bottom: '8%',  left: '34%', size: 2, c: '#cfd6e0', glow: 'rgba(200,210,230,.6)', dur: '9s',   delay: '1.4s' },
  { bottom: '20%', left: '52%', size: 3, c: '#e6eaf0', glow: 'rgba(215,225,245,.7)', dur: '8s',   delay: '2.6s' },
  { bottom: '10%', left: '68%', size: 2, c: '#cfd6e0', glow: 'rgba(200,210,230,.6)', dur: '10s',  delay: '.8s'  },
  { bottom: '16%', left: '82%', size: 3, c: '#dfe4ec', glow: 'rgba(210,220,240,.7)', dur: '7.5s', delay: '3.4s' },
  { bottom: '6%',  left: '44%', size: 2, c: '#c7cedb', glow: 'rgba(190,200,222,.6)', dur: '11s',  delay: '4.2s' },
];

export default function SwagAtmosphere() {
  const { isSwag } = useTheme();
  if (!isSwag) return null;

  return createPortal(
    <div className="swag-atmos" aria-hidden="true">
      <div className="sa-spire" style={bg('spire.png')} />
      <div className="sa-cathedral" style={bg('cathedral.png')} />
      <div className="sa-forest" style={bg('forest.png')} />
      <div className="sa-topglow" />
      <div className="sa-fog sa-fog1" />
      <div className="sa-fog sa-fog2" />
      <div className="sa-texture" style={bg('texture.png')} />
      <div className="sa-vignette" />
      <div className="sa-grain" />

      <img className="sa-watcher" src={`${A}watcher.png`} alt="" />
      <img className="sa-ornament sa-ornament--l" src={`${A}ornament.png`} alt="" />
      <img className="sa-ornament sa-ornament--r" src={`${A}ornament.png`} alt="" />

      {embers.map((e, i) => (
        <span
          key={i}
          className="sa-ember"
          style={{
            bottom: e.bottom, left: e.left,
            width: e.size, height: e.size,
            background: e.c,
            boxShadow: `0 0 ${e.size * 3}px 2px ${e.glow}`,
            animation: `emberFloat ${e.dur} ease-in ${e.delay} infinite`,
          }}
        />
      ))}

      <div className="sa-bats">
        {bats.map((b, i) => (
          <div key={i} className="sa-bat" style={b.style}>
            <svg width={b.w} height={b.h} viewBox="0 0 64 32" style={{ animation: `batFlap ${b.flap} ease-in-out infinite` }}>
              <path fill={b.fill} d={BAT_PATH} />
            </svg>
          </div>
        ))}
      </div>
    </div>,
    document.body
  );
}
