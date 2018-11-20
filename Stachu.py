from python_speech_features import mfcc
from collections import defaultdict

sample_data = [('A01M1_0_', [1,2,3,4], 48),
        ('A01M1_1_', [1, 2, 3, 4,5,6], 48),
        ('BC1M1_0_', [1, 2, 3, 4], 48),
        ('BC1M1_1_', [1, 2, 3, 4, 5, 6], 48)]


def get_params(data):
    ret = defaultdict(lambda: [])
    for el in data:
        filename = el[0].split('_')[0]
        number = el[0].split('_')[1]
        assert(len(filename) == 5 and len(number) == 1)
        mfcc_ = mfcc(el[1], samplerate=el[2], appendEnergy=True) # add paremeters
        ret[filename].append((mfcc_, number))
    return ret

