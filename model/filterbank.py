# model/filterbank.py

def run_filterbank(ts, config, method: str = "gammatone") -> Dict:
    if method == "gammatone":
        return run_gammatone(ts, config)
    elif method == "scipy":
        return run_firbank(ts, config)
    else:
        raise ValueError(f"Unknown filterbank method: {method}")

