# tests/glint_spacing_comp.py

import scipy.io
from model.echo_analyzer import linear_separate_window_10thresholds
from model.wave_params import WaveParams
import numpy as np

def test_matlab_param_file(mat_path):
    # Load the .mat file
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    # Unpack wavStructParam from .mat
    wp = data['wavStructParam']

    # Create WaveParams from loaded .mat
    wp_py = WaveParams(
        Fs=wp.Fs,
        callLenForMostFreq=wp.callLenForMostFreq,
        callLenForHighFreq=wp.callLenForHighFreq,
        callLenSpecial=wp.callLenSpecial,
        sepFlag=wp.sepFlag,
        whenBrStart=wp.whenBrStart,
        startingThPercent=wp.startingThPercent,
        th_type=str(wp.th_type),
        SepbwBRand1stEchoinSmpls=int(wp.SepbwBRand1stEchoinSmpls),
        color=str(wp.color),
        ALT=wp.ALT,
        NT=int(wp.NT),
        NoT=int(wp.NoT),
        simStruct={
            "coch": {
                "Fc": wp.simStruct.coch.Fc,
                "bmm": wp.simStruct.coch.bmm,
            }
        }
    )

    # Run the analyzer
    echo, first_gap = linear_separate_window_10thresholds(wp_py)

    print("First echo indices (binary matrix):")
    print(np.array(echo).astype(int))

    print("\nFirst gap vector (samples):")
    print(first_gap)

if __name__ == "__main__":
    test_matlab_param_file("tests/wavStructParam_test.mat")
