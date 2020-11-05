"""Routines to create filter coefficients and plot filters.

for 1st / 2nd order audio biquads

this code is included in the open educational resource
"Sascha Spors, Digital Signal Processing - Lecture notes featuring
computational examples"

hosted at
URL = ('https://github.com/spatialaudio/'
       'digital-signal-processing-lecture/tree/master/'
       'filter_design')

and is licensed under The MIT License (MIT):
Copyright (c) 2020 Sascha Spors, Frank Schultz
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy import signal


def bilinear_biquad(B, A, fs):
    """Get the bilinear transform of a 2nd-order Laplace transform.

    bilinear transform H(s)->H(z) with s=2*fs*(z-1)/(z+1)

    input:
    B[0] = B0   B[1] = B1   B[2] = B2
    A[0] = A0   A[1] = A1   A[2] = A2
    fs...sampling frequency in Hz
           Y(s)   B0*s^2+B1*s+B2   B[0]*s^2+B[1]*s+B[2]
    H(s) = ---- = -------------- = --------------------
           X(s)   A0*s^2+A1*s+A2   A[0]*s^2+A[1]*s+A[2]
    output:
    b[0] = b0   b[1] = b1   b[2] = b2
    a[0] = 1    a[1] = a1   a[2] = a2
           Y(z)   b2*z^-2+b1*z^-1+b0   b[2]*z^-2+b[1]*z^-1+b[0]
    H(z) = ---- = ------------------ = ------------------------
           X(z)   a2*z^-2+a1*z^-1+ 1   a[2]*z^-2+a[1]*z^-1+a[0]
    """
    A0, A1, A2 = A
    B0, B1, B2 = B
    fs2 = fs**2

    a0 = A2 + 2*A1*fs + 4*A0*fs2
    b0 = B2 + 2*B1*fs + 4*B0*fs2

    b1 = 2*B2 - 8*B0*fs2
    a1 = 2*A2 - 8*A0*fs2

    b2 = B2 - 2*B1*fs + 4*B0*fs2
    a2 = A2 - 2*A1*fs + 4*A0*fs2

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a


def bw_from_q(q):
    """Convert bandpass quality to bandwidth in octaves."""
    return 2/np.log(2) * np.arcsinh(1/(2*q))


def q_from_bw(bw):
    """Convert bandwidth in octaves to bandpass quality."""
    return 1 / (2*np.sinh(np.log(2)/2*bw))


def f_prewarping(f, fs):
    """Do the frequency prewarping.

    input:
    f...analog frequency in Hz to be prewarped
    fs...sampling frequency in Hz
    output:
    prewarped angular frequency in rad/s
    """
    return 2*fs*np.tan(np.pi*f/fs)


def q_prewarping(q, fm, fs, q_warp_method="cos"):
    """Do the quality prewarping.

    input:
    q...bandpass quality to be prewarped
    fm...analog mid-frequency in Hz
    fs...sampling frequency in Hz
    q_warp_method:
    "sin"...Robert Bristow-Johnson (1994): "The equivalence of various methods
    of computing biquad coefficients for audio parametric equalizers."In:
    Proc. of 97th AES Convention, San Fransisco. eq. (14)
    "cos"...Rainer Thaden (1997): "Entwicklung und Erprobung einer digitalen
    parametrischen Filterbank." Diplomarbeit, RWTH Aachen
    "tan"...Clark, R.J.; Ifeachor, E.C.; Rogers, G.M.; et al. (2000):
    "Techniques for Generating Digital Equalizer Coefficients".
    In: J. Aud. Eng. Soc. 48(4):281-298.
    output:
    prewarped quality
    """
    if q_warp_method == "sin":
        bandwidth = bw_from_q(q)
        w0 = 2*np.pi*fm / fs
        bandwidth = bandwidth*w0 / np.sin(w0)
        qpre = q_from_bw(bandwidth)
    elif q_warp_method == "cos":
        qpre = q * np.cos(np.pi*fm / fs)
    elif q_warp_method == "tan":
        qpre = q * (np.pi*fm / fs) / np.tan(np.pi*fm / fs)
    else:
        qpre = q
    return qpre


def biquad_lp1st(fc, fs):
    """Calc coeff for lowpass 1st order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    B = np.array([0., 0, 1])
    A = np.array([0, 1 / wc, 1])

    wcpre = f_prewarping(fc, fs)
    Bp = 0., 0., 1.
    Ap = 0., 1 / wcpre, 1.
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_lp2nd(fc, fs, bi=1., ai=np.sqrt(2)):
    """Calc coeff for lowpass 2nd order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    bi, ai...filter characteristics coefficients, e.g.
    bi = 0.6180, ai = 1.3617 for Bessel 2nd order
    bi = 1, ai = 1.4142 for Butterworth 2nd order (default)
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    B = np.array([0., 0, 1])
    A = np.array([bi / wc**2, ai / wc, 1])

    wcpre = f_prewarping(fc, fs)
    Bp = 0., 0., 1.
    Ap = bi / wcpre**2, ai / wcpre, 1.
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_hp1st(fc, fs):
    """Calc coeff for highpass 1st order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    B = np.array([0, 1 / wc, 0])
    A = np.array([0, 1 / wc, 1])

    wcpre = f_prewarping(fc, fs)
    Bp = 0., 1 / wcpre, 0.
    Ap = 0., 1 / wcpre, 1.
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_hp2nd(fc, fs, bi=1., ai=np.sqrt(2)):
    """Calc coeff for highpass 2nd order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    bi, ai...filter characteristics coefficients, e.g.
    bi = 0.6180, ai = 1.3617 for Bessel 2nd order
    bi = 1, ai = 1.4142 for Butterworth 2nd order (default)
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    B = np.array([1 / wc**2, 0, 0])
    A = np.array([1 / wc**2, ai / wc, bi])

    wcpre = f_prewarping(fc, fs)
    Bp = 1 / wcpre**2, 0., 0.
    Ap = 1 / wcpre**2, ai / wcpre, bi
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_bp2nd(fm, q, fs, q_warp_method="cos"):
    """Calc coeff for bandpass 2nd order.

    input:
    fm...mid frequency in Hz
    q...bandpass quality
    fs...sampling frequency in Hz
    q_warp_method..."sin", "cos", "tan"
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wm = 2*np.pi*fm
    B = np.array([0, 1 / (q*wm), 0])
    A = np.array([1 / wm**2, 1 / (q*wm), 1])

    wmpre = f_prewarping(fm, fs)
    qpre = q_prewarping(q, fm, fs, q_warp_method)
    Bp = 0., 1 / (qpre*wmpre), 0.
    Ap = 1 / wmpre**2, 1 / (qpre*wmpre), 1.
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_bs2nd(fm, q, fs, q_warp_method="cos"):
    """Calc coeff for bandstop 2nd order.

    input:
    fm...mid frequency in Hz
    q...bandpass quality
    fs...sampling frequency in Hz
    q_warp_method..."sin", "cos", "tan"
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wm = 2*np.pi*fm
    B = np.array([1 / wm**2, 0, 1])
    A = np.array([1 / wm**2, 1 / (q*wm), 1])

    wmpre = f_prewarping(fm, fs)
    qpre = q_prewarping(q, fm, fs, q_warp_method)
    Bp = 1 / wmpre**2, 0., 1.
    Ap = 1 / wmpre**2, 1 / (qpre*wmpre), 1.
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_ap1st(fc, fs, ai=1.):
    """Calc coeff for allpass 1st order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    ai...filter characteristics coefficients, e.g. ai = 1
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    B = np.array([0, -ai / wc, 1])
    A = np.array([0, +ai / wc, 1])

    wcpre = f_prewarping(fc, fs)
    Bp = 0., -ai / wcpre, 1.
    Ap = 0., +ai / wcpre, 1.
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_ap2nd(fc, fs, bi=1., ai=np.sqrt(2)):
    """Calc coeff for allpass 2nd order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    bi, ai...filter characteristics coefficients, e.g.
    bi = 1, ai = 1.4142
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    B = np.array([bi / wc**2, -ai / wc, 1])
    A = np.array([bi / wc**2, +ai / wc, 1])

    wcpre = f_prewarping(fc, fs)
    Bp = bi / wcpre**2, -ai / wcpre, 1.
    Ap = bi / wcpre**2, +ai / wcpre, 1.
    b, a = bilinear_biquad(Bp, Ap, fs)

    return B, A, b, a


def biquad_peq2nd(fm, G, q, fs, filter_type="III", q_warp_method="cos"):
    """Calc coeff for peak/dip equalizer (PEQ) 2nd order.

    input:
    fm...mid frequency in Hz
    G...gain or attenuation in dB
    q...quality
    fs...sampling frequency in Hz
    filter_type..."I", "II", "III"
    q_warp_method..."sin", "cos", "tan"
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wm = 2*np.pi*fm
    wmpre = f_prewarping(fm, fs)
    g = 10**(G/20)
    qpre = q_prewarping(q, fm, fs, q_warp_method)
    if filter_type == "I":  # aka constant-Q PEQ
        gamma = g
        delta = g
    elif filter_type == "II":  # aka symmetrical PEQ
        gamma = 1
        delta = g
    elif filter_type == 'III':  # aka one-half pad loss PEQ or midpoint PEQ
        gamma = g**0.5
        delta = g**0.5
    else:
        raise ValueError(("inappropriate filter_type, "
                         "please use 'I', 'II' or 'III' only"))
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0, 1])  # flat EQ
        A = B
        b = np.array([1., 0, 0])
        a = b
    elif G > 0:
        B = np.array([1 / wm**2, delta / (q*wm), 1])
        A = np.array([1 / wm**2, (delta/g) / (q*wm), 1])

        Bp = 1 / wmpre**2, delta / (qpre*wmpre), 1.
        Ap = 1 / wmpre**2, (delta/g) / (qpre*wmpre), 1.
        b, a = bilinear_biquad(Bp, Ap, fs)
    else:
        B = np.array([1 / wm**2, gamma / (q*wm), 1])
        A = np.array([1 / wm**2, (gamma/g) / (q*wm), 1])

        Bp = 1 / wmpre**2, gamma / (qpre*wmpre), 1.
        Ap = 1 / wmpre**2, (gamma/g) / (qpre*wmpre), 1.
        b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_peq2nd_zoelzer(fm, G, q, fs):
    """Calc coeff for peak/dip equalizer (PEQ) 2nd order.

    according to
    U. Zoelzer (2011): "DAFX - Digital Audio Effects", 2nd, Wiley, Table 2.4

    input:
    fm...mid frequency in Hz
    G...gain or attenuation in dB
    q...quality
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    K = np.tan(np.pi * fm/fs)
    V0 = 10**(G/20)
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0, 0])  # flat EQ
        a = b
    elif G > 0:
        tmp = 1 + K/q + K**2
        b = np.array([(1 + V0/q * K + K**2) / tmp,
                      2 * (K**2 - 1) / tmp,
                      (1 - V0/q * K + K**2) / tmp])
        a = np.array([1,
                      2 * (K**2 - 1) / tmp,
                     (1 - K/q + K**2) / tmp])
    else:
        tmp = 1 + K / (V0*q) + K**2
        b = np.array([(1 + K/q + K**2) / tmp,
                      2 * (K**2 - 1) / tmp,
                      (1 - K/q + K**2) / tmp])
        a = np.array([1,
                      2 * (K**2 - 1) / tmp,
                      (1 - K/(V0*q) + K**2) / tmp])
    return b, a


def biquad_peq2nd_RBJ(fm, G, q, fs):
    """Calc coeff for peak/dip equalizer (PEQ) 2nd order.

    according to
    Robert Bristow-Johnson (1994): "The equivalence of various methods of
    computing biquad coefficients for audio parametric equalizers."
    In: Proc. of 97th AES Convention, San Fransisco, eq. (16)
    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

    input:
    fm...mid frequency in Hz
    G...gain or attenuation in dB
    q...quality
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    Ksqrt = 10**(G/40)
    w0 = 2*np.pi*fm / fs
    BW = bw_from_q(q)
    gamma = np.sinh(np.log(2)/2 * (BW*w0) / np.sin(w0))*np.sin(w0)
    tmp = 1 + gamma/Ksqrt
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0, 0])  # flat EQ
        a = b
    else:
        b = np.array([(1 + gamma*Ksqrt) / tmp,
                      -2 * np.cos(w0) / tmp,
                      (1 - gamma*Ksqrt) / tmp])
        a = np.array([1,
                      -2 * np.cos(w0) / tmp,
                      (1 - gamma/Ksqrt) / tmp])
    return b, a


def biquad_lshv1st(fc, G, fs, filter_type="III"):
    """Calc coeff for lowshelving 1st order.

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    filter_type..."I", "II", "III"
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    wcpre = f_prewarping(fc, fs)
    g = 10**(G/20)
    if filter_type == "I":
        alpha = 1
    elif filter_type == "II":
        alpha = g**0.5
    elif filter_type == "III":  # one-half pad loss, midpoint
        alpha = g**0.25
    else:
        raise ValueError(("inappropriate filter_type, "
                         "please use 'I', 'II' or 'III' only"))
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0, 1])  # flat EQ
        A = B
        b = np.array([1., 0, 0])
        a = b
    elif G > 0:
        B = np.array([0, 1 / wc, g * alpha**-2])
        A = np.array([0, 1 / wc, alpha**-2])

        Bp = 0., 1 / wcpre, g * alpha**-2
        Ap = 0., 1 / wcpre, alpha**-2
        b, a = bilinear_biquad(Bp, Ap, fs)
    else:
        B = np.array([0, 1 / wc, alpha**2])
        A = np.array([0, 1 / wc, g**-1 * alpha**2])

        Bp = 0., 1 / wcpre, alpha**2
        Ap = 0., 1 / wcpre, g**-1 * alpha**2
        b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_lshv2nd(fc, G, fs,
                   filter_type="III", qz=1/np.sqrt(2), qp=1/np.sqrt(2)):
    """Calc coeff for lowshelving 2nd order.

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    filter_type..."I", "II", "III"
    qz...zero Quality, e.g. qz = 1/np.sqrt(2) for Butterworth quality
    qp...pole quality, e.g. qp = 1/np.sqrt(2) for Butterworth quality
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    g = 10**(G/20)
    wc = 2*np.pi*fc
    wcpre = f_prewarping(fc, fs)
    if filter_type == "I":
        alpha = 1
    elif filter_type == "II":
        alpha = g**0.5
    elif filter_type == "III":  # one-half pad loss, midpoint
        alpha = g**0.25
    else:
        raise ValueError(("inappropriate filter_type, "
                         "please use 'I', 'II' or 'III' only"))
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0, 1])  # flat EQ
        A = B
        b = np.array([1., 0, 0])
        a = b
    elif G > 0:
        B = np.array([1 / wc**2, g**0.5 * alpha**-1 / (qz*wc), g * alpha**-2])
        A = np.array([1 / wc**2, alpha**-1 / (qp*wc), alpha**-2])

        Bp = [1 / wcpre**2, g**0.5 * alpha**-1 / (qz*wcpre),
              g * alpha**-2]
        Ap = [1 / wcpre**2, alpha**-1 / (qp*wcpre), alpha**-2]
        b, a = bilinear_biquad(Bp, Ap, fs)
    else:
        B = np.array([1 / wc**2, alpha / (qz*wc), alpha**2])
        A = np.array([1 / wc**2, g**-0.5 * alpha / (qp*wc), g**-1 * alpha**2])

        Bp = [1 / wcpre**2, alpha / (qz*wcpre), alpha**2]
        Ap = [1 / wcpre**2, g**-0.5 * alpha / (qp*wcpre),
              g**-1 * alpha**2]
        b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_lshv2nd_Zoelzer(fc, G, fs):
    """Calc coeff for highshelving 2nd order.

    according to
    U. Zoelzer (2011): "DAFX - Digital Audio Effects", 2nd, Wiley, Table 2.3

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    V0 = 10**(G/20)
    K = np.tan(np.pi*fc / fs)
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0, 0])  # flat EQ
        a = b
    elif G > 0:
        tmp = 1 + np.sqrt(2)*K + K**2
        b = np.array([(1 + np.sqrt(2*V0)*K + V0*K**2) / tmp,
                      2 * (V0 * K**2 - 1) / tmp,
                      (1 - np.sqrt(2*V0)*K + (V0 * K**2)) / tmp])
        a = np.array([1,
                      2 * (K**2 - 1) / tmp,
                      (1 - np.sqrt(2)*K + K**2) / tmp])
    else:
        tmp = V0 + np.sqrt(2*V0)*K + K**2
        b = np.array([V0 * (1 + np.sqrt(2)*K + K**2) / tmp,
                      2*V0 * (K**2 - 1) / tmp,
                      V0 * (1 - np.sqrt(2)*K + K**2) / tmp])
        a = np.array([1,
                      2 * (K**2 - V0) / tmp,
                      (V0 - np.sqrt(2*V0)*K + K**2) / tmp])
    return b, a


def biquad_lshv2nd_RBJ(fc, G, S, fs):
    """Calc coeff for lowshelving 2nd order.

    according to
    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    S...normalized quality
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    A = 10**(G/40)
    w0 = 2*np.pi*fc / fs
    alpha = np.sin(w0)/2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0, 0])  # flat EQ
        a = b
    else:
        a0 = (A + 1) + (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha
        b = np.array([A * ((A + 1) - (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha),
                      2*A * ((A - 1) - (A + 1)*np.cos(w0)),
                      A * ((A + 1) - (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha)])
        a = np.array([a0,
                      -2 * ((A - 1) + (A + 1)*np.cos(w0)),
                      (A + 1) + (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha])
        a = a / a0
        b = b / a0
    return b, a


def biquad_hshv1st(fc, G, fs, filter_type="III"):
    """Calc coeff for highshelving 1st order.

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    filter_type..."I", "II", "III"
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    wcpre = f_prewarping(fc, fs)
    g = 10**(G/20)
    if filter_type == "I":
        alpha = 1
    elif filter_type == "II":
        alpha = g**0.5
    elif filter_type == "III":  # one-half pad loss, midpoint
        alpha = g**0.25
    else:
        raise ValueError(("inappropriate filter_type, "
                         "please use 'I', 'II' or 'III' only"))
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0, 1])  # flat EQ
        A = B
        b = np.array([1., 0, 0])
        a = b
    elif G > 0:
        B = np.array([0, g * alpha**-2 / wc, 1])
        A = np.array([0, alpha**-2 / wc, 1])

        Bp = 0., g * alpha**-2 / wcpre, 1.
        Ap = 0., alpha**-2 / wcpre, 1.
        b, a = bilinear_biquad(Bp, Ap, fs)
    else:
        B = np.array([0, alpha**2 / wc, 1])
        A = np.array([0, g**-1 * alpha**2 / wc, 1])

        Bp = 0., alpha**2 / wcpre, 1.
        Ap = 0., g**-1 * alpha**2 / wcpre, 1.
        b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_hshv2nd(fc, G, fs,
                   filter_type="III", qz=1/np.sqrt(2), qp=1/np.sqrt(2)):
    """Calc coeff for highshelving 2nd order.

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    filter_type..."I", "II", "III"
    qz...zero Quality, e.g. qz = 1/np.sqrt(2) for Butterworth quality
    qp...pole quality, e.g. qp = 1/np.sqrt(2) for Butterworth quality
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    wc = 2*np.pi*fc
    wcpre = f_prewarping(fc, fs)
    g = 10**(G/20)
    if filter_type == "I":
        alpha = 1
    elif filter_type == "II":
        alpha = g**0.5
    elif filter_type == "III":  # one-half pad loss, midpoint
        alpha = g**0.25
    else:
        raise ValueError(("inappropriate filter_type, "
                         "please use 'I', 'II' or 'III' only"))
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0, 1])  # flat EQ
        A = B
        b = np.array([1., 0, 0])
        a = b
    elif G > 0:
        B = np.array([g * alpha**-2 / wc**2, g**0.5 * alpha**-1 / (qz*wc), 1])
        A = np.array([alpha**-2 / wc**2, alpha**-1 / (qp*wc), 1])

        Bp = g * alpha**-2 / wcpre**2, g**0.5 * alpha**-1 / (qz*wcpre), 1.
        Ap = alpha**-2 / wcpre**2, alpha**-1 / (qp*wcpre), 1.
        b, a = bilinear_biquad(Bp, Ap, fs)
    else:
        B = np.array([alpha**2 / wc**2, alpha / (qz*wc), 1])
        A = np.array([g**-1 * alpha**2 / wc**2, g**-0.5 * alpha / (qp*wc), 1])

        Bp = alpha**2 / wcpre**2, alpha / (qz*wcpre), 1.
        Ap = g**-1 * alpha**2 / wcpre**2, g**-0.5 * alpha/(qp*wcpre), 1.
        b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_hshv2nd_Zoelzer(fc, G, fs):
    """Calc coeff for highshelving 2nd order.

    according to
    U. Zoelzer (2011): "DAFX - Digital Audio Effects", 2nd, Wiley, Table 2.3

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    V0 = 10**(G/20)
    K = np.tan(np.pi*fc / fs)
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0, 0])  # flat EQ
        a = b
    elif G > 0:
        tmp = 1 + np.sqrt(2)*K + K**2
        b = np.array([(V0 + np.sqrt(2*V0)*K + K**2) / tmp,
                      2 * (K**2 - V0) / tmp,
                      (V0 - np.sqrt(2*V0)*K + K**2) / tmp])
        a = np.array([1, 2 * (K**2 - 1) / tmp,
                      (1 - np.sqrt(2)*K + K**2) / tmp])
    else:
        tmp = 1 + np.sqrt(2*V0)*K + (V0 * K**2)
        b = np.array([V0 * (1 + np.sqrt(2)*K + K**2) / tmp,
                      2*V0 * (K**2 - 1) / tmp,
                      V0 * (1 - np.sqrt(2.)*K + K**2) / tmp])
        a = np.array([1, 2 * (V0 * K**2 - 1) / tmp,
                      (1 - np.sqrt(2*V0)*K + (V0 * K**2)) / tmp])
    return b, a


def biquad_hshv2nd_RBJ(fc, G, S, fs):
    """Calc coeff for highshelving 2nd order.

    according to
    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    S...normalized quality
    fs...sampling frequency in Hz
    output:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    """
    A = 10**(G/40)
    w0 = 2*np.pi*fc / fs
    alpha = np.sin(w0)/2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0, 0])  # flat EQ
        a = b
    else:
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        b = np.array([A * ((A + 1) + (A - 1) * np.cos(w0)
                      + 2 * np.sqrt(A) * alpha),
                      - 2 * A * ((A - 1) + (A + 1) * np.cos(w0)),
                      A * ((A + 1) + (A - 1) * np.cos(w0)
                      - 2 * np.sqrt(A) * alpha)])
        a = np.array([a0,
                      2 * ((A - 1) - (A + 1) * np.cos(w0)),
                      (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha])
        b = b / a0
        a = a / a0
    return b, a


def zplane_plot(ax, z, p, k):
    """Plot pole/zero/gain plot of discrete-time, linear-time-invariant system.

    Note that the for-loop handling might be not very efficient
    for very long FIRs

    z...array of zeros in z-plane
    p...array of poles in z-zplane
    k...gain factor

    taken from own work
    URL = ('https://github.com/spatialaudio/signals-and-systems-exercises/'
           'blob/master/sig_sys_tools.py')

    currently we don't use the ax input parameter, we rather just plot
    in hope for getting an appropriate place for it from the calling function
    """
    # draw unit circle
    Nf = 2**7
    Om = np.arange(Nf) * 2*np.pi/Nf
    plt.plot(np.cos(Om), np.sin(Om), 'C7')

    try:  # TBD: check if this pole is compensated by a zero
        circle = Circle((0, 0), radius=np.max(np.abs(p)),
                        color='C7', alpha=0.15)
        plt.gcf().gca().add_artist(circle)
    except ValueError:
        print('no pole at all, ROC is whole z-plane')

    zu, zc = np.unique(z, return_counts=True)  # find and count unique zeros
    for zui, zci in zip(zu, zc):  # plot them individually
        plt.plot(np.real(zui), np.imag(zui), ms=7,
                 color='C0', marker='o', fillstyle='none')
        if zci > 1:  # if multiple zeros exist then indicate the count
            plt.text(np.real(zui), np.imag(zui), zci)

    pu, pc = np.unique(p, return_counts=True)  # find and count unique poles
    for pui, pci in zip(pu, pc):  # plot them individually
        plt.plot(np.real(pui), np.imag(pui), ms=7,
                 color='C3', marker='x')
        if pci > 1:  # if multiple poles exist then indicate the count
            plt.text(np.real(pui), np.imag(pui), pci)

    plt.text(0, +1, 'k=%f' % k)
    plt.text(0, -1, 'ROC for causal: white')
    plt.axis('square')
    # plt.axis([-2, 2, -2, 2])
    plt.xlabel(r'$\Re\{z\}$')
    plt.ylabel(r'$\Im\{z\}$')
    plt.grid(True)


def bode_plot(B, A, b, a, fs, N, fig=None):
    """Realize a bode plot containing magnitude, phase and zplane.

    input:
    B...numerator coefficients Laplace transfer function
    A...denominator coefficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator coefficients z-transfer function
    fs...sampling frequency in Hz
    output:
    bode plot as new figure
    """
    if fig is None:
        fig = plt.figure()
    z, p, k = signal.tf2zpk(b, a)
    W, Hd = signal.freqz(b, a, N)
    s, Ha = signal.freqs(B, A, fs*W)
    if Hd[0] == 0:
        Hd[0] = 1e-15  # avoid zero at DC for plotting dB
    if Ha[0] == 0:
        Ha[0] = 1e-15
    f = fs*W / (2*np.pi)

    gs = fig.add_gridspec(2, 2)
    # magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(f, 20*np.log10(np.abs(Ha)), "C0",
             label=r'$|H(\omega)|$ continuous-time',
             linewidth=3)
    ax1.plot(f, 20*np.log10(np.abs(Hd)), "C1",
             label=(r'$|H(\Omega)|$ discrete-time, fs=%5.f Hz' % fs),
             linewidth=2)
    ax1.set_xscale("log")
    ax1.set_yscale("linear")
    ax1.set_xlabel(r'$f$ / Hz', color="xkcd:navy blue")
    ax1.set_ylabel(r'$A$ / dB', color="xkcd:navy blue")
    ax1.set_title("Bode plot: magnitude", color="xkcd:navy blue")
    ax1.set_xlim(20, 20000)
    ax1.set_xticks((20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000))
    ax1.set_xticklabels(["20", "50",
                         "100", "200", "500",
                         "1k", "2k", "5k",
                         "10k", "20k"], color="xkcd:navy blue")
    ax1.set_ylim(-15, 15)
    ax1.set_yticks(np.arange(-15, 15+3, 3))
    ax1.set_yticklabels(["-15", "-12", "-9", "-6", "-3", "0",
                         "3", "6", "9", "12", "15"],
                        color="xkcd:navy blue")
    ax1.legend(loc="best")
    ax1.grid(True, which="both", axis="both",
             linestyle="-", linewidth=0.5, color=(0.8, 0.8, 0.8))

    # phase
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(f, (np.angle(Ha)*180/np.pi), "C0",
             label=r'$\mathrm{angle}(H('r'\omega))$ continuous-time',
             linewidth=3)
    ax2.plot(f, (np.angle(Hd)*180/np.pi), 'C1',
             label=(r'$\mathrm{angle}(H(\Omega))$ discrete-time, '
                    'fs=%5.f Hz' % fs),
             linewidth=2)
    ax2.set_xscale("log")
    ax2.set_yscale("linear")
    ax2.set_xlabel(r'$f$ / Hz', color="xkcd:navy blue")
    ax2.set_ylabel(r'$\phi$ / deg', color="xkcd:navy blue")
    ax2.set_title("Bode plot: phase", color="xkcd:navy blue")
    ax2.set_xlim(20, 20000)
    ax2.set_xticks((20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000))
    ax2.set_xticklabels(["20", "50",
                         "100", "200", "500",
                         "1k", "2k", "5k",
                         "10k", "20k"],
                        color="xkcd:navy blue")
    ax2.set_ylim(-180, +180)
    ax2.set_yticks(np.arange(-180, 180+45, 45))
    ax2.set_yticklabels(["-180", "-135", "-90", "-45", "0",
                         "45", "90", "135", "180"],
                        color="xkcd:navy blue")
    ax2.legend(loc="best")
    ax2.grid(True, which="both", axis="both",
             linestyle="-", linewidth=0.5, color=(0.8, 0.8, 0.8))

    # zplane
    ax3 = fig.add_subplot(gs[:, 1])
    zplane_plot(ax3, z, p, k)

    print("B =", B)
    print("A =", A)
    print("b =", b)
    print("a =", a)


def magnitude_plot_overlay(x, y, title, legend, fig=None):
    """Realize a bode plot containing magnitude for overlay."""
    if fig is None:
        plt.figure()
    sz = y.shape
    lines = plt.semilogx(x, 20*np.log10(np.abs(y)))
    plt.legend(lines[:sz[1]], legend)
    plt.autoscale("tight")
    plt.title(title)
    plt.xlabel(r'$f$ / Hz')
    plt.ylabel(r'20 log10 |$H$| / dB')
    plt.axis([1000, 24000, 0, 18])
    plt.yticks(np.arange(-18, 18+3, 3))
    plt.xticks((20, 50, 100, 200, 500,
                1000, 2000, 5000, 10000, 20000),
               ["20", "50", "100", "200", "500",
                "1k", "2k", "5k", "10k", "20k"])
    plt.xlim(20, x.max())
    plt.ylim(-18, 18)
    plt.grid(True, which="both", axis="both",
             linestyle="-", linewidth=0.5, color=(0.8, 0.8, 0.8))
    plt.show()
