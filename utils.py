import torch
from torch import nn
import numpy as np


EPSILON = 1e-2


def linear_quantize(samples, q_levels):
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return samples.long()


def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1


def q_zero(q_levels):
    return q_levels // 2


# Antonio Bonafonte
# Add ulaw quantizer

MU = 255.
LOG_MU1 = 5.5451774444795623    # log(1+MU)


def ulaw(x, max_value=1.0):
    v = MU/max_value
    y = x.sign() * (v * x.abs() +1.).log()/LOG_MU1
    return y    


def iulaw(c, max_value=1.0, mu=255.):
    x = (c.abs() * LOG_MU1).exp() - 1
    y = c.sign() * x/MU
    return y


EPSILONs = 1e-6


def midrise(x, q_levels=256):
    x = 0.5 * (x+1.0)
    x *= (q_levels - EPSILONs)
    return x.long()


def imidrise(xq, q_levels=256):
    return xq.float() * 2.0 / q_levels - 1.0

        
def uquantize(samples,q_levels):
    return midrise(ulaw(samples), q_levels)


def udequantize(samples, q_levels):
    return iulaw(imidrise(samples,q_levels))

