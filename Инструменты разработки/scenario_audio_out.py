#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сценарий 3.8: наушники подключили во время чтения - звук переходит в них.

На Mac подключение наушников - это смена устройства вывода системы. Сценарий
делает то же самое программно: создаёт временное составное устройство поверх
динамиков (система разрешает сделать его устройством вывода; звук остаётся в
динамиках), во время звука потоком делает его устройством системы, проверяет
по CoreAudio, что играет уже оно, возвращает как было и удаляет устройство.
Звук - едва слышный тон.

    python "Инструменты разработки/scenario_audio_out.py"
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from core import audio_out as A
from core.stream import SoundSink

if sys.platform != "darwin":
    print("только Mac: на Windows поток идёт через «Microsoft Sound Mapper», его Windows "
          "переводит сама")
    sys.exit(0)
import ctypes
import objc
from Foundation import NSDictionary, NSNumber

ca = A._coreaudio()
ca.AudioHardwareCreateAggregateDevice.restype = ctypes.c_int32
ca.AudioHardwareDestroyAggregateDevice.restype = ctypes.c_int32


def uid(dev):
    a = A._Addr(A._cc("uid "), A._cc("glob"), 0)
    p, size = ctypes.c_void_p(), ctypes.c_uint32(8)
    ca.AudioObjectGetPropertyData(dev, ctypes.byref(a), 0, None, ctypes.byref(size), ctypes.byref(p))
    return str(objc.objc_object(c_void_p=p.value))


start = A.system_default()
u = uid(start)
agg = NSDictionary.dictionaryWithDictionary_({
    "uid": "herald.check.aggregate", "name": "Herald проверка", "subdevices": [{"uid": u}],
    "master": u, "private": NSNumber.numberWithInt_(0)})
dev = ctypes.c_uint32(0)
if ca.AudioHardwareCreateAggregateDevice(ctypes.c_void_p(objc.pyobjc_id(agg)), ctypes.byref(dev)):
    print("не создалось временное устройство - нечем проверить")
    sys.exit(0)
other = dev.value
time.sleep(1.0)
sr = 24000
tone = (0.003 * np.sin(2 * np.pi * 440 * np.arange(int(sr * 6)) / sr)).astype(np.float32)
ok = True
sink = SoundSink()
try:
    sink.open(sr)
    sink.write(tone)
    sink.finish()
    time.sleep(1.0)
    on_start = A.running_somewhere(start)
    A.set_system_default(other)                 # «подключили наушники»
    time.sleep(1.5)
    moved = A.running_somewhere(other) and sink.switches == 1
    pos = sink.position
    A.set_system_default(start)                 # «отключили»
    time.sleep(1.5)
    back = A.running_somewhere(start) and sink.switches == 2
    sink.done.wait(10)
    print(f"  играло на устройстве системы: {on_start}")
    print(f"  после смены устройства системы - на новом: {moved} (позиция {pos:.1f} с)")
    print(f"  после возврата - снова на прежнем: {back}")
    print(f"  доиграло до конца: {sink.done.is_set()}, сыграно {sink.position:.1f} с из 6,0")
    ok = on_start and moved and back and sink.done.is_set() and sink.position > 5.5
finally:
    if A.system_default() != start:
        A.set_system_default(start)
    sink.stop()
    ca.AudioHardwareDestroyAggregateDevice(other)
print("всё в порядке" if ok else "ПРОВАЛЕНО")
sys.exit(0 if ok else 1)
