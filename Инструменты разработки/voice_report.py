#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Глубокий разбор голоса по всем значимым параметрам, ориентир — живой диктор.

Диктор (тот, из кого клонируют) — эталон-цель. Для каждой записи считаем набор
акустических параметров и показываем отклонение. Не нужно быть звукорежиссёром:
чем ближе к диктору, тем натуральнее.

ГРОМКОСТЬ:  LUFS, RMS, пик, крест-фактор (динамика), LRA, клиппинг
ТЕМБР:      7 полос %, центроид (яркость), наклон спектра
АРТЕФАКТЫ:  металл (плоскость 4-11к), шипение (5-9к), бочка (300-700),
            HNR, гул сети 50Гц, ТРЕСК (щелчки/с), ШЕРОХОВАТОСТЬ (дребезг),
            ДЖИТТЕР (нестабильность тона -> «скрип/фрай»)
ГОЛОС:      F0, живость интонации, озвученность

Запуск:
    python3 voice_report.py --target ДИКТОР.mp3 файл1.mp3 файл2.mp3 ...
    python3 voice_report.py --json out.json --gate --target ... ...
С --gate возвращает код !=0, если у кандидата металл/шипение/треск заметно
выше диктора (защита от «железного/шипящего/скрипящего» на автомате).
"""
import argparse, json, subprocess, re
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
    p = subprocess.run(["ffmpeg","-v","info","-ss","1","-t","45","-i",path,
                        "-af","ebur128","-f","null","-"], capture_output=True, text=True)
    I=re.findall(r"I:\s*(-?\d+\.?\d*)\s*LUFS", p.stderr)
    L=re.findall(r"LRA:\s*(-?\d+\.?\d*)\s*LU", p.stderr)
    return (float(I[-1]) if I else float("nan"), float(L[-1]) if L else float("nan"))

def _mag(y): return np.abs(np.fft.rfft(y*np.hanning(len(y)))), np.fft.rfftfreq(len(y),1/SR)

def bands(y):
    S,f=_mag(y); P=S**2; t=P.sum()+1e-12
    e=[(20,60),(60,120),(120,250),(250,500),(500,2000),(2000,5000),(5000,16000)]
    return [100*P[(f>=lo)&(f<hi)].sum()/t for lo,hi in e]

def centroid(y):
    S,f=_mag(y); return float((f*S).sum()/(S.sum()+1e-9))

def tilt(y):
    S,f=_mag(y); m=(f>=100)&(f<8000)
    return float(np.polyfit(np.log2(f[m]+1e-9), 20*np.log10(S[m]+1e-9), 1)[0])

def flatness(y,lo,hi):
    n=2048;hop=512;w=np.hanning(n);f=np.fft.rfftfreq(n,1/SR);m=(f>=lo)&(f<hi);v=[]
    for i in range(0,len(y)-n,hop):
        S=np.abs(np.fft.rfft(y[i:i+n]*w))**2;b=S[m]+1e-10
        v.append(np.exp(np.mean(np.log(b)))/np.mean(b))
    return float(np.median(v)) if v else float("nan")

def sibilance(y):
    S,f=_mag(y);P=S**2
    return float(P[(f>=5000)&(f<9000)].sum()/(P[(f>=1000)&(f<4000)].sum()+1e-9))

def boxy(y):
    S,f=_mag(y);b=S[(f>=300)&(f<700)];return float(np.max(b)/(np.median(b)+1e-9))

def hnr(y):
    n=1024;hop=512;v=[]
    for i in range(0,len(y)-n,hop):
        fr=y[i:i+n]*np.hanning(n)
        if np.sqrt(np.mean(fr**2))<0.01: continue
        ac=np.correlate(fr,fr,'full')[n-1:];ac=ac/(ac[0]+1e-9)
        seg=ac[int(SR/400):int(SR/70)]
        if not len(seg): continue
        pk=min(float(np.max(seg)),0.999);v.append(10*np.log10((pk+1e-9)/(1-pk+1e-9)))
    return float(np.median(v)) if v else float("nan")

def hum(y):
    S,f=_mag(y);P=S**2;t=P.sum()+1e-12;e=0
    for h in (50,100,150,200): e+=P[(f>=h-3)&(f<=h+3)].sum()
    return float(100*e/t)

def crackle(y):
    """Щелчки/треск: резкие скачки сэмплов, событий в секунду."""
    d=np.abs(np.diff(y)); med=np.median(d)+1e-9
    clicks=int(np.sum(d> max(0.06, 25*med)))
    return float(clicks/(len(y)/SR))

def roughness(y):
    """Дребезг/жёсткость: энергия огибающей в 30-150 Гц, % (сенсорная шероховатость)."""
    env=np.abs(y).astype(np.float64); env=env-np.mean(env)
    E=np.abs(np.fft.rfft(env*np.hanning(len(env)))); f=np.fft.rfftfreq(len(env),1/SR)
    tot=E[(f>1)&(f<300)].sum()+1e-9
    return float(100*E[(f>=30)&(f<150)].sum()/tot)

def clip(y): return float(100*np.mean(np.abs(y)>0.99))
def level(y):
    rms=np.sqrt(np.mean(y**2))+1e-12;pk=np.max(np.abs(y))+1e-12
    return 20*np.log10(rms),20*np.log10(pk)-20*np.log10(rms)

def pitch(y):
    try:
        import librosa
        y22=librosa.resample(y.astype(np.float32),orig_sr=SR,target_sr=22050)
        f0,vf,_=librosa.pyin(y22,sr=22050,fmin=70,fmax=400,frame_length=2048)
        v=f0[~np.isnan(f0)]
        if len(v)<5: return float("nan"),float("nan"),float("nan"),float("nan")
        semis=12*np.log2(v/np.median(v))
        jit=100*np.median(np.abs(np.diff(v))/v[:-1])       # джиттер -> «скрип/фрай»
        return float(np.median(v)),float(np.std(semis)),float(100*np.mean(vf>0.5)),float(jit)
    except Exception:
        return (float("nan"),)*4

def analyze(path):
    y=decode(path)
    I,LRA=lufs_lra(path); rms,crest=level(y); cl=clip(y)
    yb=y/(np.max(np.abs(y))+1e-9)
    b=bands(yb); f0,f0sd,vf,jit=pitch(yb)
    return dict(lufs=I,lra=LRA,rms=rms,crest=crest,clip=cl,
        sub=b[0],bass=b[1],low=b[2],lowmid=b[3],mid=b[4],presence=b[5],air=b[6],
        centroid=centroid(yb),tilt=tilt(yb),
        metal=flatness(yb,4000,11000),sibilance=sibilance(yb),boxy=boxy(yb),
        hnr=hnr(yb),hum=hum(yb),crackle=crackle(yb),rough=roughness(yb),jitter=jit,
        f0=f0,f0_range=f0sd,voiced=vf)

ORDER=[("lufs","LUFS громк"),("lra","LRA"),("crest","крест/динам"),("clip","клип%"),
    ("bass","бас%"),("lowmid","нижсер%"),("mid","сер%"),("presence","присут%"),
    ("air","воздух%"),("centroid","центроид"),("tilt","наклон"),
    ("metal","МЕТАЛЛ"),("sibilance","ШИПЕНИЕ"),("boxy","БОЧКА"),
    ("crackle","ТРЕСК/с"),("rough","ШЕРОХОВ%"),("jitter","ДЖИТТЕР%"),
    ("hnr","HNR"),("hum","гул50"),("f0","F0"),("f0_range","живость"),("voiced","озвуч%")]

def resolve_source(path):
    """Запись чтеца по пути из <голос>.src.json — даже если папку переименовали.

    Папку с образцами уже раз переименовывали («Мои книги» -> «Образцы голосов»),
    и сверка молча выключилась. Если по старому пути файла нет, ищем тот же файл
    по имени внутри проекта.
    """
    import os
    if os.path.exists(path):
        return path
    root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name=os.path.basename(path)
    for d,_dirs,files in os.walk(root):
        if ".venv" in d or ".git" in d:
            continue
        if name in files:
            return os.path.join(d,name)
    return path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target"); ap.add_argument("--voice",
        help="имя голоса: запись его чтеца найдётся сама (по <голос>.src.json)")
    ap.add_argument("--json"); ap.add_argument("--gate",action="store_true")
    ap.add_argument("files",nargs="+"); a=ap.parse_args()
    # Сверять надо со СВОИМ чтецом. Один раз эталон был взят от другого диктора,
    # и все выводы поехали: у тёмного голоса цель 2500 Гц яркости, у звонкого
    # 3300, и общий рецепт обработки портит обоих. Имя голоса само приводит к
    # нужной записи — угадывать больше не нужно.
    if a.voice and not a.target:
        import os
        src=os.path.expanduser(f"~/.cache/text2audio/voices/{a.voice}.src.json")
        if not os.path.exists(src):
            print(f"нет {src}: неизвестно, из какой записи сделан голос «{a.voice}»"); return 2
        info=json.load(open(src,encoding="utf-8")); a.target=resolve_source(info["source"])
        print(f"эталон: {a.target}" + (f"  (читает {info['reader']})" if info.get("reader") else ""))
    if not a.target:
        print("нужен --target ФАЙЛ или --voice ИМЯ"); return 2
    t=analyze(a.target); t["_name"]="ОРИГИНАЛ"; rows=[t]
    for p in a.files:
        r=analyze(p); r["_name"]=p.rsplit("/",1)[-1]; rows.append(r)
    hdr=f"{'параметр':13s} "+" ".join(f"{r['_name'][:15]:>15s}" for r in rows)
    print(hdr); print("-"*len(hdr))
    for k,lbl in ORDER:
        line=f"{lbl:13s} "
        for r in rows:
            v=r.get(k,float('nan')); line+=f"{v:15.2f} " if v==v else f"{'--':>15s} "
        print(line)
    if a.json: json.dump(rows,open(a.json,"w"),ensure_ascii=False,indent=1); print(f"\nJSON: {a.json}")
    if a.gate:
        # Порог 1,4x по металлу и шипению, а не 1,25/1,3. Замерено на четырёх
        # прогонах одного текста: «металл» гуляет 0,06-0,08 от запуска к запуску
        # при ОДНОМ образце — каждый синтез стартует со своего шума. Прежний
        # порог лежал внутри этого разброса и браковал исправный звук. Грубый
        # брак виден всё равно: у отозванных вариантов было 1,6-2,2x.
        bad=[r["_name"] for r in rows[1:]
             if r["metal"]>t["metal"]*1.4 or r["sibilance"]>t["sibilance"]*1.4
             or r["crackle"]>t["crackle"]*2.0 or r["rough"]>t["rough"]*1.4]
        if bad: print("НЕ ПРОШЛИ (шип/металл/треск выше диктора):", ", ".join(bad)); return 1
    return 0

if __name__=="__main__":
    raise SystemExit(main())
