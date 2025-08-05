# config_loader.py

from pathlib import Path
from typing import List
import yaml
import polars as pl
from pydantic import BaseModel, Field
from model.target import Target

CONFIG_PATH = Path("config/config.yaml")
SCENARIO_PATH = Path("config/scenarios/uniform_10.csv")


# ---------- Config Dataclasses ----------

class WaveConfig(BaseModel):
    sample_rate: int
    sound_speed: float
    wavelength: float


class BinauralConfig(BaseModel):
    ear_separation: float
    head_width: float
    max_turn_rate: float
    min_turn_rate: float
    update_rate: int
    max_iterations: int
    convergence_threshold: float
    file_normalize: int
    file_normlevel: int
    file_timeseries: int
    file_specgram: int
    file_wignerville: int
    coch_panel: str
    coch_ord: int
    coch_ripple: int
    coch_bw: int
    coch_fmin: int
    coch_fmax: int
    coch_steps: int
    coch_fcenter: int
    coch_mode: int
    coch_tfplot: int
    coch_grpdelay: int
    coch_bmmplot: int
    neur_panel: str
    neur_probplot: int
    neur_isihist: int
    neur_rasterplot: int
    neur_rcf_rect: int
    neur_rcf_comp: int
    neur_rcf_filt: int
    neur_rcf_spike: int
    neur_rcf_ord: int
    neur_rcf_fc: int
    neur_rcf_thrlev: str
    neur_rcf_refaper: int
    neur_rcf_refrper: int
    neur_rcf_reffcn: int
    neur_rcf_freqresp: int
    neur_biol_parm: List[List[float]]
    neur_biol_absper: float
    neur_biol_relper: float
    neur_biol_cr: float
    neur_biol_plotstates: int
    neur_rand_mode: int
    neur_rand_rate: int
    neur_rand_amod: int
    neur_rand_rper: str
    neur_rand_fanout: int
    neur_rand_seed: str
    cn_ntot: int
    cn_nsyn: int
    cn_nrec: int
    cn_wrec: int
    cn_lifparams: List[List[float]]
    cn_plottran: int
    cn_autocorr: int
    cn_plotvmem: int
    cn_isihist: int
    cn_rasterplot: int


class Config(BaseModel):
    wave: WaveConfig
    binaural: BinauralConfig

# ---------- Loaders ----------

def load_config(path: Path = CONFIG_PATH) -> Config:
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def load_scenarios(path: Path = SCENARIO_PATH) -> List[Target]:
    df = pl.read_csv(path)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    return [Target(**row) for row in df.to_dicts()]
