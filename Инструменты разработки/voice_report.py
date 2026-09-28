#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Глубокий разбор голоса по всем значимым параметрам, с ориентиром на ОРИГИНАЛ.

Идея простая: живой диктор (тот, из кого клонируют) — эталон-цель. Для каждой
записи считаем набор акустических параметров и показываем ОТКЛОНЕНИЕ от диктора.
Не нужно быть звукорежиссёром: чем ближе к диктору, тем «натуральнее/богаче».

Что меряем и что это значит на слух:
  ГРОМКОСТЬ
    LUFS         — воспринимаемая громкость (стандарт вещания). Тихо = ниже.
    RMS дБ       — средний уровень сигнала.
    пик дБ       — максимум; близко к 0 = риск искажений.
    крест-фактор — пик минус RMS; мало = пересжато (лимитер), «неживо».
    LRA          — разброс громкости по EBU; мало = плоско/задавлено.
  СПЕКТР (тембр)
    7 полос %    — распределение энергии от суб-баса до «воздуха».
    центроид Гц  — «яркость»: выше = звонче/резче, ниже = глуше.
    наклон       — общий баланс тёмный/светлый (дБ на октаву).
  АРТЕФАКТЫ
    металл       — плоскость спектра 4-11 кГц (шум вокодера). Выше = «железо».
    шипение      — доля энергии 5-9 кГц к 1-4 кГц. Выше = свистящие «с/ш».
    бочка        — резонанс 300-700 Гц (пик/медиана). Выше = гулко/«из бочки».
    HNR дБ       — гармоники к шуму. Выше = чище голос.
    гул сети     — энергия 50 Гц и кратных. Выше = фон/наводка.
    клиппинг %   — доля сэмплов у потолка. Выше нуля = перегруз.
  ГОЛОС
    F0 Гц        — средняя высота тона.
    F0 разброс   — живость интонации (полутона). Мало = монотонно.
    озвуч. %     — доля озвученной речи.

Запуск:
    python3 voice_report.py --target ДИКТОР.mp3 запись1.mp3 запись2.mp3 ...
    python3 voice_report.py --json out.json --target ... ...
"""
import argparse, json, subprocess, sys, re
import numpy as np

SR = 48000

def decode(path, offset=1.0, dur=45.0):
    cmd = ["ffmpeg", "-v", "error"]
    if offset: cmd += ["-ss", str(offset)]
    if dur: cmd += ["-t", str(dur)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)

def lufs_lra(path):
    """Интегральная громкость и LRA через ffmpeg ebur128."""
    p = subprocess.run(["ffmpeg","-v","info","-ss","1","-t","45","-i",path,"-af","ebur128",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = p.stderr
    I = re.findall(r"I:\s*(-?\d+\.?\d*)\s*LUFS", txt)
    LRA = re.findall(r"LRA:\s*(-?\d+\.?\d*)\s*LU", txt)
    return (float(I[-1]) if I else float("nan"),
            float(LRA[-1]) if LRA else float("nan"))

def _rfft_mag(y):
    return np.abs(np.fft.rfft(y * np.hanning(len(y)))), np.fft.rfftfreq(len(y), 1/SR)

def band_pcts(y):
    S, f = _rfft_mag(y); P = S**2; tot = P.sum() + 1e-12
    edges = [(20,60,"суб"),(60,120,"бас"),(120,250,"низ"),(250,500,"нижсер"),
             (500,2000,"сер"),(2000,5000,"присут"),(5000,16000,"воздух")]
    return {name: 100*P[(f>=lo)&(f<hi)].sum()/tot for lo,hi,name in edges}

def centroid(y):
    S, f = _rfft_mag(y); return float((f*S).sum()/(S.sum()+1e-9))

def tilt(y):
    S, f = _rfft_mag(y); m=(f>=100)&(f<8000)
    lf=np.log2(f[m]+1e-9); lp=20*np.log10(S[m]+1e-9)
    return float(np.polyfit(lf, lp, 1)[0])   # дБ/октава

def flatness(y, lo, hi):
    n=2048; hop=512; w=np.hanning(n); f=np.fft.rfftfreq(n,1/SR); m=(f>=lo)&(f<hi)
    vals=[]
    for i in range(0,len(y)-n,hop):
        S=np.abs(np.fft.rfft(y[i:i+n]*w))**2; b=S[m]+1e-10
        vals.append(np.exp(np.mean(np.log(b)))/np.mean(b))
    return float(np.median(vals)) if vals else float("nan")

def sibilance(y):
    S,f=_rfft_mag(y); P=S**2
    hi=P[(f>=5000)&(f<9000)].sum(); mid=P[(f>=1000)&(f<4000)].sum()+1e-9
    return float(hi/mid)

def boxy(y):
    S,f=_rfft_mag(y); b=S[(f>=300)&(f<700)]
    return float(np.max(b)/(np.median(b)+1e-9))

def hnr(y):
    n=1024; hop=512; vals=[]
    for i in range(0,len(y)-n,hop):
        fr=y[i:i+n]*np.hanning(n)
        if np.sqrt(np.mean(fr**2))<0.01: continue
        ac=np.correlate(fr,fr,'full')[n-1:]; ac=ac/(ac[0]+1e-9)
        seg=ac[int(SR/400):int(SR/70)]
        if not len(seg): continue
        pk=min(float(np.max(seg)),0.999)
        vals.append(10*np.log10((pk+1e-9)/(1-pk+1e-9)))
    return float(np.median(vals)) if vals else float("nan")

def hum(y):
    S,f=_rfft_mag(y); P=S**2; tot=P.sum()+1e-12; e=0
    for h in (50,100,150,200):
        e+=P[(f>=h-3)&(f<=h+3)].sum()
    return float(100*e/tot)

def level(y):
    rms=np.sqrt(np.mean(y**2))+1e-12; pk=np.max(np.abs(y))+1e-12
    return 20*np.log10(rms), 20*np.log10(pk), 20*np.log10(pk)-20*np.log10(rms)

def clip_pct(y):
    return float(100*np.mean(np.abs(y)>0.99))

def noise_floor(y):
    """Уровень самых тихих 10% кадров — прокси шумовой полки/шипения в паузах."""
    n=int(0.1*SR); r=[np.sqrt(np.mean(y[i:i+n]**2)) for i in range(0,len(y)-n,n)]
    r=np.array(r)+1e-9
    return float(20*np.log10(np.percentile(r,10)))

def pitch(y):
    try:
        import librosa
        y22=librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=22050)
        f0,vf,_=librosa.pyin(y22, sr=22050, fmin=70, fmax=400, frame_length=2048)
        v=f0[~np.isnan(f0)]
        if len(v)<5: return float("nan"),float("nan"),float("nan")
        semis=12*np.log2(v/np.median(v))
        return float(np.median(v)), float(np.std(semis)), float(100*np.mean(vf>0.5))
    except Exception:
        return float("nan"),float("nan"),float("nan")

def analyze(path):
    y=decode(path)
    if y.size: y=y  # уровень меряем как есть (важно для «тихо»)
    I,LRA=lufs_lra(path)
    rms,pk,crest=level(y)
    yb=y/(np.max(np.abs(y))+1e-9)  # для спектра нормируем — форма не зависит от уровня
    b=band_pcts(yb)
    f0,f0sd,vf=pitch(yb)
    return dict(
        lufs=I, lra=LRA, rms=rms, peak=pk, crest=crest, clip=clip_pct(y),
        noise=noise_floor(yb),
        sub=b["суб"],bass=b["бас"],low=b["низ"],lowmid=b["нижсер"],mid=b["сер"],
        presence=b["присут"],air=b["воздух"],
        centroid=centroid(yb), tilt=tilt(yb),
        metal=flatness(yb,4000,11000), sibilance=sibilance(yb), boxy=boxy(yb),
        hnr=hnr(yb), hum=hum(yb), f0=f0, f0_range=f0sd, voiced=vf,
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--json")
    ap.add_argument("files", nargs="+")
    a=ap.parse_args()
    tgt=analyze(a.target); tgt["_name"]="ОРИГИНАЛ (диктор)"
    rows=[tgt]
    for p in a.files:
        r=analyze(p); r["_name"]=p.rsplit("/",1)[-1]; rows.append(r)
    order=[("lufs","LUFS громк"),("lra","LRA"),("crest","крест"),("peak","пик дБ"),
           ("clip","клип%"),("noise","шум.полка"),
           ("bass","бас%"),("lowmid","нижсер%"),("mid","сер%"),("presence","присут%"),
           ("air","воздух%"),("centroid","центроид"),("tilt","наклон"),
           ("metal","МЕТАЛЛ"),("sibilance","ШИПЕНИЕ"),("boxy","БОЧКА"),("hnr","HNR"),
           ("hum","гул50"),("f0","F0"),("f0_range","F0разб"),("voiced","озвуч%")]
    w=max(len(r["_name"]) for r in rows)
    hdr=f"{'параметр':12s} "+" ".join(f"{r['_name'][:16]:>16s}" for r in rows)
    print(hdr); print("-"*len(hdr))
    for key,lbl in order:
        line=f"{lbl:12s} "
        for r in rows:
            v=r.get(key,float('nan'))
            line+=f"{v:16.2f} " if v==v else f"{'--':>16s} "
        print(line)
    if a.json:
        json.dump(rows, open(a.json,"w"), ensure_ascii=False, indent=1)
        print(f"\nJSON: {a.json}")

if __name__=="__main__":
    main()
