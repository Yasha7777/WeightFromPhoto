import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';

/* ============================================================
   ThemeProvider — тема «обычный формат» ⇄ «свага»
   ------------------------------------------------------------
   • mode ('normal' | 'swag') — сохраняется в localStorage.
     Дефолт — 'swag' (по брифу свага = режим по умолчанию).
   • data-theme выставляется на <html> → swag.css перекрашивает
     весь интерфейс, включая глобальную шапку сайта.
   • flip(target) — переключение с анимацией «разлома»:
       0мс    — поднимаем флаг flipping (чёрная заливка + тряска);
       470мс  — фактически меняем тему (под чёрным экраном);
       960мс  — снимаем флаг.
   • intro — вуаль при КАЖДОМ заходе (если режим swag и не включён
     prefers-reduced-motion). Гаснет сама через 3.4с или по клику.
   ============================================================ */

const ThemeCtx = createContext(null);
export const useTheme = () => useContext(ThemeCtx);

const STORAGE_KEY = 'kh-theme';
const prefersReduced =
  typeof window !== 'undefined' &&
  window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const readInitialMode = () => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'normal' || saved === 'swag') return saved;
  } catch (_) {}
  return 'swag';
};

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState(readInitialMode);
  const [flipping, setFlipping] = useState(false);
  const [intro, setIntro] = useState(() => mode === 'swag' && !prefersReduced);

  // зеркала актуальных значений для синхронного чтения в flip()
  const modeRef = useRef(mode);
  const flippingRef = useRef(false);
  const timers = useRef([]);

  useEffect(() => { modeRef.current = mode; }, [mode]);

  // отражаем тему на <html> + сохраняем выбор
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode);
    try { localStorage.setItem(STORAGE_KEY, mode); } catch (_) {}
  }, [mode]);

  // авто-закрытие интро-вуали
  useEffect(() => {
    if (!intro) return;
    const t = setTimeout(() => setIntro(false), 3400);
    return () => clearTimeout(t);
  }, [intro]);

  // чистим хвостовые таймеры разлома при размонтировании
  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const skipIntro = useCallback(() => setIntro(false), []);

  const flip = useCallback((target) => {
    if (flippingRef.current || modeRef.current === target) return;
    flippingRef.current = true;
    setFlipping(true);
    const t1 = setTimeout(() => { modeRef.current = target; setMode(target); }, 470);
    const t2 = setTimeout(() => { flippingRef.current = false; setFlipping(false); }, 960);
    timers.current.push(t1, t2);
  }, []);

  const value = {
    mode,
    isSwag: mode === 'swag',
    flipping,
    intro,
    skipIntro,
    flip,
    goNormal: () => flip('normal'),
    goSwag: () => flip('swag'),
    reducedMotion: prefersReduced,
  };

  return <ThemeCtx.Provider value={value}>{children}</ThemeCtx.Provider>;
}
