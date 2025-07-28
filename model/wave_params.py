# model/wave_params.py

from dataclasses import dataclass
from typing import Literal

@dataclass
class WaveParams:
    Fs: int = 500_000                          # sampling frequency
    callLenForMostFreq: float = 0.5e-3         # .5 ms
    callLenForHighFreq: float = 1e-3           # 1 ms
    callLenSpecial: float = 1.8e-3             # 1.8 ms
    sepFlag: int = 1                           # 1 = fixed separation
    whenBrStart: float = 0.5e-3                # 0.5 ms
    startingThPercent: int = 3                 # threshold start percent
    th_type: Literal["const", "dynamic"] = "const"
    SepbwBRand1stEchoinSmpls: int = 5000       # ~10ms at 500kHz
    color: str = "b"
    ALT: int = -25                             # amplitude latency trading
    NT: int = 10                               # number of thresholds
    NoT: int = 1                                # current threshold (to be iterated)
    simStruct: dict = None                      # to be set per signal

