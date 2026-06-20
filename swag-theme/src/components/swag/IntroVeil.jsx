import { createPortal } from 'react-dom';
import { useTheme } from '../../theme/ThemeProvider';

/* ============================================================
   IntroVeil — «пробуждающийся глаз» при каждом заходе.
   Показывается только в режиме «свага» и только пока intro=true
   (ThemeProvider сам гасит через 3.4с и пропускает при
   prefers-reduced-motion). Клик по вуали — пропустить.
   ============================================================ */

const A = `${import.meta.env.BASE_URL}swag/`;

export default function IntroVeil() {
  const { isSwag, intro, skipIntro } = useTheme();
  if (!isSwag || !intro) return null;

  return createPortal(
    <div className="swag-intro" onClick={skipIntro}>
      <div className="si-spire" style={{ backgroundImage: `url(${A}spire.png)` }} />
      <div className="si-glow" />
      <div className="si-ring si-ring--1" />
      <div className="si-ring si-ring--2" />
      <img className="si-watcher" src={`${A}watcher.png`} alt="" />
      <div className="si-crack" />
      <div className="si-titles">
        <div className="si-title">КАРЕЛИЯ СТРОЙ</div>
        <div className="si-sub">✝ ОТ ТРАНСИЛЬВАНИИ ДО ЛЕСОВ ОРЕГОНА ✝</div>
      </div>
      <div className="si-hint">нажмите, чтобы войти</div>
    </div>,
    document.body
  );
}
