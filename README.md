# 📦 Volume Estimation from Images

![n8n](https://img.shields.io/badge/n8n-automation-orange)
![Supabase](https://img.shields.io/badge/Supabase-DB-green)
![CLIP](https://img.shields.io/badge/CLIP-embeddings-blue)
![LLM](https://img.shields.io/badge/LLM-analysis-purple)

Оценка **объёма и массы сыпучих материалов** по фото.

## 🚀 Pipeline
Фото → CLIP → Supabase → LLM → Fusion → 📊 результат

## 📊 Output
- Объём (м³)  
- Масса (т)  
- 95% интервал  
- Confidence режим  

## 🧠 Особенности
- Probabilistic fusion (AI + DB)  
- Учёт неопределённости (σ)  
- Устойчивость к шуму  

## ⚠️ Ограничения
- Нет segmentation  
- Нет точного масштаба  