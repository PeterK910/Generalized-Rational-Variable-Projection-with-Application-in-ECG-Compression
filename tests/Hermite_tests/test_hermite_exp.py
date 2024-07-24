from Hermite.hermite_exp import quant
from Hermite.hermite_exp import hermite_exp
import numpy as np
import pytest
import struct

@pytest.mark.parametrize(
    "data,exp1,exp2",
    [
        ([0.010074,0.00026887], 0.007843, 0.0),
        ([-0.040827,0.052044], -0.039216, 0.054902),
    ],
)
def test_quant(data, exp1, exp2):    
    data=np.array(data)
    eps=8

    res=quant(data, eps)
    assert round(res[0], 6)==exp1
    assert round(res[1], 6)==exp2

def test_1():
    pass
"""
beat=np.array([0.42971,0.3989,0.3989,0.35268,0.3989,0.36808,0.46053,0.50675,0.50675,0.58379,0.58379,0.55297,0.50675,0.52216,0.52216,0.50675,0.53756,0.55297,0.63001,0.55297,0.6146,0.63001,0.64542,0.6146,0.59919,0.58379,0.6146,0.58379,0.52216,0.56838,0.63001,0.6146,0.64542,0.66082,0.63001,0.67623,0.56838,0.55297,0.52216,0.49134,0.49134,0.44512,0.47594,0.50675,0.50675,0.55297,0.49134,0.47594,0.49134,0.47594,0.46053,0.44512,0.46053,0.50675,0.49134,0.53756,0.59919,0.56838,0.6146,0.6146,0.53756,0.52216,0.46053,0.44512,0.42971,0.46053,0.44512,0.47594,0.47594,0.46053,0.44512,0.46053,0.42971,0.38349,0.3989,0.41431,0.33727,0.3989,0.3989,0.41431,0.3989,0.42971,0.46053,0.42971,0.42971,0.36808,0.33727,0.36808,0.33727,0.29105,0.36808,0.38349,0.3989,0.46053,0.44512,0.41431,0.38349,0.3989,0.35268,0.36808,0.33727,0.3989,0.44512,0.47594,0.53756,0.53756,0.55297,0.58379,0.6146,0.64542,0.70705,0.75327,0.90734,1.0152,1.0768,0.99978,0.76867,0.35268,-0.10954,-0.57176,-1.0956,-1.6041,-2.1433,-2.8366,-3.53,-4.1463,-4.7471,-5.1323,-5.3018,-5.4251,-5.4559,-5.5021,-5.4867,-5.4867,-5.4867,-5.4251,-5.2402,-5.0091,-4.6547,-4.362,-4.0692,-3.7611,-3.5146,-3.2372,-2.9291,-2.7288,-2.5285,-2.3436,-2.0817,-1.8506,-1.5578,-1.2651,-1.034,-0.71043,-0.32524,-0.017098,0.26023,0.46053,0.53756,0.50675,0.53756,0.55297,0.66082,0.72245,0.86112,0.92275,1.046,1.0768,1.0306,1.0922,1.0306,1.046,1.0768,1.0768,1.1384,1.1693,1.2771,1.3233,1.3079,1.3079,1.3542,1.3079,1.3233,1.3542,1.3696,1.4158,1.4466,1.5544,1.5853,1.7085,1.7393,1.8164,1.8318,1.878,1.9088,1.9088,1.9396,2.0783,2.1553,2.2478,2.294,2.4018,2.5097,2.5251,2.5713,2.5251,2.5405,2.5251,2.4943,2.5405,2.5559,2.5713,2.633,2.71,2.8178,2.9257,2.9873,2.9257,3.0027,2.9873,3.049,3.1568,3.203,3.2492,3.3879,3.3571,3.4033,3.3263,3.2492,3.203,3.1568,3.1106,3.0798,3.0181,2.9873,2.9565,2.8641,2.8487,2.6946,2.6176,2.4943,2.3556,2.2478,2.0937,1.9704,1.8318,1.7701,1.7085,1.5853,1.5236,1.385,1.2925,1.1847,1.0922,0.93815,0.90734,0.8149,0.76867,0.70705,0.75327,0.70705,0.72245,0.69164,0.67623,0.64542,0.63001,0.53756,0.52216,0.47594,0.50675,0.52216,0.55297,0.55297,0.63001,0.63001,0.59919,0.64542,0.58379,0.56838,0.55297,0.55297,0.52216,0.56838,0.56838,0.6146,0.64542,0.66082,0.64542,0.63001,0.6146,0.59919,0.52216,0.50675,0.52216,0.55297,0.58379,0.58379,0.63001,0.67623,0.64542,0.59919,0.58379,0.6146,0.52216,0.58379,0.52216,0.56838,0.6146,0.6146,0.64542,0.63001,0.59919,0.6146,0.50675,0.49134,0.49134,0.50675,0.46053,0.47594,0.53756,0.55297,0.59919,0.58379,0.56838,0.53756,0.50675,0.52216,0.42971,0.42971,0.3989,0.42971,0.46053,0.46053,0.52216,0.49134,0.44512,0.42971,0.41431,0.41431,0.38349,0.36808,0.36808,0.41431,0.42971,0.44512,0.46053,0.44512,0.47594,0.44512,0.3989,0.3989,0.32186,0.36808,0.33727,0.35268,0.38349,0.38349,0.41431,0.41431,0.32186,0.38349,0.33727,0.27564,0.30646,0.30646,0.30646,0.38349,0.36808,0.38349,0.38349,0.41431,0.41431,0.33727,0.32186,0.36808,0.30646,0.29105,0.27564,0.30646,0.30646,0.35268,0.36808,0.33727,0.36808,0.33727,0.32186,0.32186,0.29105,0.29105,0.29105,0.29105,0.35268,0.38349,0.41431,0.41431,0.38349,0.38349,0.33727,0.32186,0.27564,0.26023,0.22942,0.27564,0.26023,0.32186,0.35268,0.29105,0.35268,0.35268,0.30646,0.29105,0.30646,0.22942,0.29105,0.24483,0.32186,0.32186,0.35268,0.32186,0.32186,0.29105,0.24483,0.22942,0.15238,0.1986,0.1832,0.16779,0.1832,0.29105,0.30646,0.30646,0.32186,0.29105,0.30646,0.21401,0.16779,0.15238,0.16779,0.1832,0.21401,0.22942,0.1986,0.27564,0.26023,0.26023,0.1986,0.1986,0.12157,0.090753,0.10616,0.1832,0.22942,0.29105,0.30646,0.33727,0.29105,0.35268,0.26023,0.21401,0.16779,0.1832,0.21401,0.24483,0.22942,0.27564,0.32186,0.29105,0.22942,0.29105,0.22942,0.22942,0.15238,0.16779,0.1832,0.27564])
onsets=112
offsets=139
basenums=np.array([2,7,6])
acc=8
lowerbound=0.1
upperbound=65
step=45/50

aprx, bm, la, co, b, prds = hermite_exp(beat, onsets, offsets, basenums, acc, lowerbound, upperbound, step)
print(aprx)
print(bm)
print(la)
print(co)
"""