"""routines to create filter coefficients and plot of
1st / 2nd order audio biquads

URL = ('https://github.com/spatialaudio/'
       'digital-signal-processing-lecture/tree/master/'
       'filter_design')
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle


def bilinear_biquad(B, A, fs=48000):
    """get the bilinear transform of a 2nd-order Laplace transform

    bilinear transform H(s)->H(z) with s=2*fs*(z-1)/(z+1)

    input:
    B[0]=B0   B[1]=B1   B[2]=B2
    A[0]=A0   A[1]=A1   A[2]=A2
    fs...sampling frequency in Hz (default 48000)

           Y(s)   B0*s^2+B1*s+B2   B[0]*s^2+B[1]*s+B[2]
    H(s) = ---- = -------------- = --------------------
           X(s)   A0*s^2+A1*s+A2   A[0]*s^2+A[1]*s+A[2]

    output:
    b[0]=b0   b[1]=b1   b[2]=b2
    a[0]=1    a[1]=a1   a[2]=a2

           Y(z)   b2*z^-2+b1*z^-1+b0   b[2]*z^-2+b[1]*z^-1+b[0]
    H(z) = ---- = ------------------ = ------------------------
           X(z)   a2*z^-2+a1*z^-1+ 1   a[2]*z^-2+a[1]*z^-1+a[0]
    """
    a = np.array([0., 0., 0.])
    b = np.array([0., 0., 0.])
    fs2 = fs**2.

    tmp = A[2] + 2*A[1]*fs + 4*A[0]*fs2
    b[0] = B[2] + 2*B[1]*fs + 4*B[0]*fs2

    b[1] = 2*B[2] - 8*B[0]*fs2
    a[1] = 2*A[2] - 8*A[0]*fs2

    b[2] = B[2] - 2*B[1]*fs + 4*B[0]*fs2
    a[2] = A[2] - 2*A[1]*fs + 4*A[0]*fs2

    a = a / tmp  # normalize such that a[0]=1
    b = b / tmp
    a[0] = 1.  # set as double

    return b, a


def bw_from_q(QBP):
    """convert bandpass quality to bandwidth in octaves"""
    return 2/np.log(2) * np.arcsinh(1/(2*QBP))


def q_from_bw(BWoct):
    """convert bandwidth in ocatves to bandpass quality"""
    return 1 / (2*np.sinh(np.log(2)/2*BWoct))


def prewarping_f(f, fs=48000):
    """do the frequency prewarping

    input:
    f...analog frequency in Hz to be prewarped
    fs...sampling frequency in Hz

    output:
    prewarped digital angular frequency in rad/s
    """
    return 2*fs*np.tan(np.pi*f/fs)


def prewarping_q(Q, fm, fs=48000, WarpType="cos"):
    """do the quality prewarping

    input:
    Q...bandpass quality to be prewarped
    fm...analog mid-frequency in Hz
    fs...sampling frequency in Hz
    WarpType:
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
    if WarpType == "sin":
        BW = bw_from_q(Q)
        w0 = 2*np.pi*fm / fs
        BW = BW*w0 / np.sin(w0)
        Qp = q_from_bw(BW)
    elif WarpType == "cos":
        Qp = Q * np.cos(np.pi*fm / fs)
    elif WarpType == "tan":
        Qp = Q * (np.pi*fm / fs) / np.tan(np.pi*fm / fs)
    else:
        Qp = Q
    return Qp


def biquad_lp1st(fc, fs):
    """calc coeff for lowpass 1st order

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    B = np.array([0.,
                  0.,
                  1.])
    A = np.array([0.,
                  1. / wc,
                  1.])
    Bp = np.array([0.,
                   0.,
                   1.])
    Ap = np.array([0.,
                   1. / wp,
                   1.])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_lp2nd(fc, fs, bi, ai):
    """calc coeff for lowpass 2nd order

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    bi, ai...filter characteristics coefficients, e.g.
    bi = 0.6180, ai = 1.3617 for Bessel
    bi = 1, ai = 1.4142 for Butterworth
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    B = np.array([0.,
                  0.,
                  1.])
    A = np.array([bi / (wc**2),
                  ai / wc,
                  1.])
    Bp = np.array([0.,
                   0.,
                   1.])
    Ap = np.array([bi / (wp**2),
                   ai / wp,
                   1.])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_hp1st(fc, fs):
    """calc coeff for highpass 1st order

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    B = np.array([0.,
                  1. / wc,
                  0.])
    A = np.array([0.,
                  1. / wc,
                  1.])
    Bp = np.array([0.,
                   1. / wp,
                   0.])
    Ap = np.array([0.,
                   1. / wp,
                   1.])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_hp2nd(fc, fs, bi, ai):
    """calc coeff for highpass 2nd order

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    bi, ai...filter characteristics coefficients, e.g.
    bi = 0.6180, ai = 1.3617 for Bessel
    bi = 1, ai = 1.4142 for Butterworth
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    B = np.array([1. / (wc**2),
                  0.,
                  0.])
    A = np.array([1. / (wc**2),
                  ai / wc,
                  bi])
    Bp = np.array([1. / (wp**2),
                   0.,
                   0.])
    Ap = np.array([1. / (wp**2),
                   ai / wp,
                   bi])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_bp2nd(fm, Q, fs, QWarpType):
    """calc coeff for bandpass 2nd order

    input:
    fm...mid frequency in Hz
    Q...bandpass quality
    fs...sampling frequency in Hz
    QWarpType..."sin", "cos", "tan"
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wm = 2*np.pi*fm
    wp = prewarping_f(fm, fs)
    Qp = prewarping_q(Q, fm, fs, QWarpType)
    B = np.array([0.,
                  1. / (Q*wm),
                  0.])
    A = np.array([1. / (wm**2),
                  1. / (Q*wm),
                  1.])
    Bp = np.array([0.,
                   1. / (Qp*wp),
                   0.])
    Ap = np.array([1. / (wp**2),
                   1. / (Qp*wp),
                   1.])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_bs2nd(fm, Q, fs, QWarpType):
    """calc coeff for bandstop 2nd order

    input:
    fm...mid frequency in Hz
    Q...bandpass quality
    fs...sampling frequency in Hz
    QWarpType..."sin", "cos", "tan"
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wm = 2*np.pi*fm
    wp = prewarping_f(fm, fs)
    Qp = prewarping_q(Q, fm, fs, QWarpType)
    B = np.array([1. / wm**2,
                  0.,
                  1.])
    A = np.array([1. / wm**2,
                  1. / (Q*wm),
                  1.])
    Bp = np.array([1. / wp**2,
                   0.,
                   1.])
    Ap = np.array([1. / wp**2,
                   1. / (Qp*wp),
                   1.])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_ap1st(fc, ai, fs):
    """calc coeff for allpass 1st order

    input:
    fc...cut frequency in Hz
    ai...filter characteristics coefficients, e.g. ai = 1
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    B = np.array([0.,
                  -ai / wc,
                  1.])
    A = np.array([0.,
                  +ai / wc,
                  1.])
    Bp = np.array([0.,
                   -ai / wp,
                   1.])
    Ap = np.array([0.,
                   +ai / wp,
                   1.])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_ap2nd(fc, bi, ai, fs):
    """calc coeff for allpass 2nd order

    input:
    fc...cut frequency in Hz
    bi, ai...filter characteristics coefficients, e.g.
    bi = 1, ai = 1.4142
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    B = np.array([bi / (wc**2),
                  -ai / wc,
                  1.])
    A = np.array([bi / (wc**2),
                  +ai / wc,
                  1.])
    Bp = np.array([bi/(wp**2.),
                   -ai/wp,
                   1.])
    Ap = np.array([bi/(wp**2.),
                   +ai/wp,
                   1.])
    b, a = bilinear_biquad(Bp, Ap, fs)
    return B, A, b, a


def biquad_peq2nd(fm, G, Q, fs, PEQType, QWarpType):
    """calc coeff for peak/dip equalizer (PEQ) 2nd order

    input:
    fm...mid frequency in Hz
    G...gain or attenuation in dB
    Q...quality
    fs...sampling frequency in Hz
    PEQType..."I", "II", "III"
    QWarpType..."sin", "cos", "tan"
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wm = 2*np.pi*fm
    wp = prewarping_f(fm, fs)
    g = 10**(G/20)
    Qp = prewarping_q(Q, fm, fs, QWarpType)
    if PEQType == "I":  # aka constant-Q PEQ
        gamma = g
        delta = g
    elif PEQType == "II":  # aka symmetrical PEQ
        gamma = 1.
        delta = g
    elif PEQType == 'III':  # aka one-half pad loss PEQ
        gamma = g**0.5
        delta = g**0.5
    else:
        gamma = unknown_PEQType
        delta = unknown_PEQType
    if G > 0.:
        B = np.array([1. / wm**2,
                      delta / (Q*wm),
                      1.])
        A = np.array([1. / wm**2,
                      (delta/g) / (Q*wm),
                      1.])
        Bp = np.array([1. / wp**2,
                       delta / (Qp*wp),
                       1.])
        Ap = np.array([1. / wp**2,
                       (delta/g) / (Qp*wp),
                       1.])
    else:
        B = np.array([1. / wm**2,
                      gamma / (Q*wm),
                      1.])
        A = np.array([1. / wm**2,
                      (gamma/g) / (Q*wm),
                      1.])
        Bp = np.array([1. / wp**2,
                       gamma / (Qp*wp),
                       1.])
        Ap = np.array([1. / wp**2,
                       (gamma/g) / (Qp*wp),
                       1.])
    b, a = bilinear_biquad(Bp, Ap, fs)

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0., 1.])
        A = np.array([0., 0., 1.])
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])

    return B, A, b, a


def biquad_peq2nd_zoelzer(fm, G, Q, fs):
    """calc coeff for peak/dip equalizer (PEQ) 2nd order according to

    U. Zoelzer (2011): "DAFX - Digital Audio Effects", 2nd, Wiley, Table 2.4

    input:
    fm...mid frequency in Hz
    G...gain or attenuation in dB
    Q...quality
    fs...sampling frequency in Hz

    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    K = np.tan(np.pi * fm/fs)
    V0 = 10**(G/20)
    b = np.array([1., 0., 0.])
    a = np.array([1., 0., 0.])
    if G > 0.:
        tmp = 1 + K/Q + K**2
        b[0] = (1 + V0/Q * K + K**2) / tmp
        b[1] = 2 * (K**2 - 1) / tmp
        b[2] = (1 - V0/Q * K + K**2) / tmp
        a[1] = 2 * (K**2 - 1) / tmp
        a[2] = (1 - K/Q + K**2) / tmp
    else:
        tmp = 1 + K / (V0*Q) + K**2
        b[0] = (1 + K/Q + K**2) / tmp
        b[1] = 2 * (K**2 - 1) / tmp
        b[2] = (1 - K/Q + K**2) / tmp
        a[1] = 2 * (K**2 - 1) / tmp
        a[2] = (1 - K/(V0*Q) + K**2) / tmp

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])

    return b, a


def biquad_peq2nd_RBJ(fm, G, Q, fs):
    """calc coeff for peak/dip equalizer (PEQ) 2nd order according to

    Robert Bristow-Johnson (1994): "The equivalence of various methods of
    computing biquad coefficients for audio parametric equalizers."
    In: Proc. of 97th AES Convention, San Fransisco, eq. (16)
    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

    input:
    fm...mid frequency in Hz
    G...gain or attenuation in dB
    Q...quality
    fs...sampling frequency in Hz

    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    b = np.array([1., 0., 0.])
    a = np.array([1., 0., 0.])
    Ksqrt = 10**(G/40)
    w0 = 2*np.pi*fm / fs
    BW = bw_from_q(Q)
    gamma = np.sinh(np.log(2)/2 * (BW*w0) / np.sin(w0))*np.sin(w0)
    tmp = 1 + gamma/Ksqrt

    b[0] = (1 + gamma*Ksqrt) / tmp
    b[1] = -2 * np.cos(w0) / tmp
    b[2] = (1 - gamma*Ksqrt) / tmp
    a[0] = 1.
    a[1] = -2 * np.cos(w0) / tmp
    a[2] = (1 - gamma/Ksqrt) / tmp

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])

    return b, a


def biquad_lshv1st(fc, G, fs, ShvType):
    """calc coeff for lowshelving 1st order

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    ShvType..."I", "II", "III"
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    g = 10**(G/20)
    if ShvType == "I":
        alpha = 1.
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if G > 0.:
        B = np.array([0.,
                      1. / wc,
                      g * alpha**-2])
        A = np.array([0.,
                      1. / wc,
                      alpha**-2])
        Bp = np.array([0.,
                       1. / wp,
                       g * alpha**-2])
        Ap = np.array([0.,
                       1. / wp,
                       alpha**-2])
    else:
        B = np.array([0.,
                      1. / wc,
                      alpha**2])
        A = np.array([0.,
                      1. / wc,
                      g**-1 * alpha**2])
        Bp = np.array([0.,
                       1. / wp,
                       alpha**2])
        Ap = np.array([0.,
                       1. / wp,
                       g**-1 * alpha**2])
    b, a = bilinear_biquad(Bp, Ap, fs)

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0., 1.])
        A = np.array([0., 0., 1.])
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])

    return B, A, b, a


def biquad_lshv2nd(fc, G, Qz, Qp, fs, ShvType):
    """calc coeff for lowshelving 2nd order

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    Qz...zero Quality, e.g. Qz = 1./np.sqrt(2.) for Butterworth quality
    Qp...pole quality, e.g. Qp = 1./np.sqrt(2.) for Butterworth quality
    fs...sampling frequency in Hz
    ShvType..."I", "II", "III"
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    g = 10**(G/20)
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    if ShvType == "I":
        alpha = 1.
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if G > 0.:
        B = np.array([1. / wc**2,
                      g**0.5 * alpha**-1 / (Qz*wc),
                      g * alpha**-2])
        A = np.array([1. / wc**2,
                      alpha**-1 / (Qp*wc),
                      alpha**-2])
        Bp = np.array([1. / wp**2,
                       g**0.5 * alpha**-1 / (Qz*wp),
                       g * alpha**-2])
        Ap = np.array([1. / wp**2,
                       alpha**-1 / (Qp*wp),
                       alpha**-2])
    else:
        B = np.array([1. / wc**2,
                      alpha / (Qz*wc),
                      alpha**2])
        A = np.array([1. / wc**2,
                      g**-0.5 * alpha / (Qp*wc),
                      g**-1 * alpha**2])
        Bp = np.array([1. / wp**2,
                       alpha / (Qz*wp),
                       alpha**2])
        Ap = np.array([1. / wp**2,
                       g**-0.5 * alpha / (Qp*wp),
                       g**-1 * alpha**2])
    b, a = bilinear_biquad(Bp, Ap, fs)

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0., 1.])
        A = np.array([0., 0., 1.])
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])

    return B, A, b, a


def biquad_lshv2nd_Zoelzer(fc, G, fs):
    """calc coeff for highshelving 2nd order according to

    U. Zoelzer (2011): "DAFX - Digital Audio Effects", 2nd, Wiley, Table 2.3

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    V0 = 10**(G/20)
    K = np.tan(np.pi*fc / fs)
    b = np.array([1., 0., 0.])
    a = np.array([1., 0., 0.])
    if G > 0.:
        tmp = 1 + np.sqrt(2)*K + K**2
        b[0] = (1 + np.sqrt(2*V0)*K + V0*K**2) / tmp
        b[1] = 2 * (V0 * K**2 - 1) / tmp
        b[2] = (1 - np.sqrt(2*V0)*K + (V0 * K**2)) / tmp
        a[1] = 2 * (K**2 - 1) / tmp
        a[2] = (1 - np.sqrt(2)*K + K**2) / tmp
    else:
        tmp = V0 + np.sqrt(2*V0)*K + K**2
        b[0] = V0 * (1 + np.sqrt(2)*K + K**2) / tmp
        b[1] = 2*V0 * (K**2 - 1) / tmp
        b[2] = V0 * (1 - np.sqrt(2)*K + K**2) / tmp
        a[1] = 2 * (K**2 - V0) / tmp
        a[2] = (V0 - np.sqrt(2*V0)*K + K**2) / tmp

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])
    return b, a


def biquad_lshv2nd_RBJ(fc, G, S, fs):
    """calc coeff for lowshelving 2nd order according to

    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    S...normalized quality
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    b = np.array([1., 0., 0.])
    a = np.array([1., 0., 0.])
    A = 10**(G/40)
    w0 = 2*np.pi*fc / fs
    alpha = np.sin(w0)/2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    b[0] = A * ((A + 1) - (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b[1] = 2*A * ((A - 1) - (A + 1)*np.cos(w0))
    b[2] = A * ((A + 1) - (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a[0] = (A + 1) + (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a[1] = -2 * ((A - 1) + (A + 1)*np.cos(w0))
    a[2] = (A + 1) + (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha
    b = b / a[0]
    a = a / a[0]
    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])
    return b, a


def biquad_hshv1st(fc, G, fs, ShvType):
    """calc coeff for highshelving 1st order

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    ShvType..."I", "II", "III"
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    g = 10**(G/20)
    if ShvType == "I":
        alpha = 1.
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if G > 0.:
        B = np.array([0.,
                      g * alpha**-2 / wc,
                      1.])
        A = np.array([0.,
                      alpha**-2 / wc,
                      1.])
        Bp = np.array([0.,
                       g * alpha**-2 / wp,
                       1.])
        Ap = np.array([0.,
                       alpha**-2 / wp,
                       1.])
    else:
        B = np.array([0.,
                      alpha**2 / wc,
                      1.])
        A = np.array([0.,
                      g**-1 * alpha**2 / wc,
                      1.])
        Bp = np.array([0.,
                       alpha**2 / wp,
                       1.])
        Ap = np.array([0.,
                       g**-1 * alpha**2 / wp,
                       1.])
    b, a = bilinear_biquad(Bp, Ap, fs)

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0., 1.])
        A = np.array([0., 0., 1.])
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])

    return B, A, b, a


def biquad_hshv2nd(fc, G, Qz, Qp, fs, ShvType):
    """calc coeff for highshelving 2nd order

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    Qz...zero Quality, e.g. Qz = 1./np.sqrt(2.) for Butterworth quality
    Qp...pole quality, e.g. Qp = 1./np.sqrt(2.) for Butterworth quality
    fs...sampling frequency in Hz
    ShvType..."I", "II", "III"
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wp = prewarping_f(fc, fs)
    g = 10**(G/20)
    if ShvType == "I":
        alpha = 1.
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if G > 0.:
        B = np.array([g * alpha**-2 / wc**2,
                      g**0.5 * alpha**-1 / (Qz*wc),
                      1.])
        A = np.array([alpha**-2 / wc**2,
                      alpha**-1 / (Qp*wc),
                      1.])
        Bp = np.array([g * alpha**-2 / wp**2,
                       g**0.5 * alpha**-1 / (Qz*wp),
                       1.])
        Ap = np.array([alpha**-2 / wp**2,
                       alpha**-1 / (Qp*wp),
                       1.])
    else:
        B = np.array([alpha**2 / wc**2,
                      alpha / (Qz*wc),
                      1.])
        A = np.array([g**-1 * alpha**2 / wc**2,
                      g**-0.5 * alpha / (Qp*wc),
                      1.])
        Bp = np.array([alpha**2 / wp**2,
                       alpha / (Qz*wp),
                       1.])
        Ap = np.array([g**-1 * alpha**2 / wp**2,
                       g**-0.5 * alpha/(Qp*wp),
                       1.])
    b, a = bilinear_biquad(Bp, Ap, fs)

    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        B = np.array([0., 0., 1.])
        A = np.array([0., 0., 1.])
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])

    return B, A, b, a


def biquad_hshv2nd_Zoelzer(fc, G, fs):
    """calc coeff for highshelving 2nd order according to

    U. Zoelzer (2011): "DAFX - Digital Audio Effects", 2nd, Wiley, Table 2.3

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    V0 = 10**(G/20)
    K = np.tan(np.pi*fc / fs)
    b = np.array([1., 0., 0.])
    a = np.array([1., 0., 0.])
    if G > 0.:
        tmp = 1 + np.sqrt(2)*K + K**2
        b[0] = (V0 + np.sqrt(2*V0)*K + K**2) / tmp
        b[1] = 2 * (K**2 - V0) / tmp
        b[2] = (V0 - np.sqrt(2*V0)*K + K**2) / tmp
        a[1] = 2 * (K**2 - 1) / tmp
        a[2] = (1 - np.sqrt(2)*K + K**2) / tmp
    else:
        tmp = 1 + np.sqrt(2*V0)*K + (V0 * K**2)
        b[0] = V0 * (1 + np.sqrt(2)*K + K**2) / tmp
        b[1] = 2*V0 * (K**2 - 1) / tmp
        b[2] = V0 * (1 - np.sqrt(2.)*K + K**2) / tmp
        a[1] = 2 * (V0 * K**2 - 1) / tmp
        a[2] = (1 - np.sqrt(2*V0)*K + (V0 * K**2)) / tmp
    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])
    return b, a


def biquad_hshv2nd_RBJ(fc, G, S, fs):
    """calc coeff for highshelving 2nd order according to

    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    S...normalized quality
    fs...sampling frequency in Hz
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    b = np.array([1., 0., 0.])
    a = np.array([1., 0., 0.])
    A = 10**(G/40)
    w0 = 2*np.pi*fc / fs
    alpha = np.sin(w0)/2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    b[0] = A * ((A + 1) + (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b[1] = -2*A * ((A - 1) + (A + 1)*np.cos(w0))
    b[2] = A * ((A + 1) + (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a[0] = (A + 1) - (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a[1] = 2 * ((A - 1) - (A + 1)*np.cos(w0))
    a[2] = (A + 1) - (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha
    b = b / a[0]
    a = a / a[0]
    # if G==0 dB make filter flat
    if np.isclose(G, 0., rtol=1e-05, atol=1e-08, equal_nan=False):
        b = np.array([1., 0., 0.])
        a = np.array([1., 0., 0.])
    return b, a


def zplane_plot(ax, z, p):
    """realize a zplane plot

    input:
    ax...axes handle
    z...zeros
    p...poles
    output:
    zplane plot into ax
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(np.real(z), np.imag(z),
            "o", label="zeros",
            color="C2", fillstyle="none",
            markersize=15, markeredgewidth=3)
    ax.plot(np.real(p), np.imag(p),
            "x", label="poles",
            color="C3", fillstyle="none",
            markersize=15, markeredgewidth=3)
    ax.axvline(0, color="0.7")
    ax.axhline(0, color="0.7")
    unit_circle = Circle((0, 0), radius=1, fill=False,
                         color="black", linestyle="-", alpha=0.9)
    ax.add_patch(unit_circle)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel(r'Real{$z$}', color="xkcd:navy blue")
    ax.set_ylabel(r'Imag{$z$}', color="xkcd:navy blue")
    ax.set_title("Poles x and zeros o of discrete-time domain filter",
                 color="xkcd:navy blue")
    ax.axis("equal")
    ax.set_xlim(-1.25, +1.25)
    ax.set_xticks(np.arange(-1.25, 1.25+0.25, 0.25))
    ax.set_xticklabels(["-1.25", "-1", "-0.75", "-0.5", "-0.25", "0",
                        "0.25", "0.5", "0.75", "1", "1.25"],
                       color="xkcd:navy blue")
    ax.set_ylim(-1.25, +1.25)
    ax.set_yticks(np.arange(-1.25, 1.25+0.25, 0.25))
    ax.set_yticklabels(["-1.25", "-1", "-0.75", "-0.5", "-0.25", "0",
                        "0.25", "0.5", "0.75", "1", "1.25"],
                       color="xkcd:navy blue")
    ax.legend(loc="best")
    ax.grid(True, which="both", axis="both",
            linestyle="-", linewidth=0.5, color=(0.8, 0.8, 0.8))


def bode_plot(fig, B, A, b, a, fs, N=2**12):
    """realize a bode plot containing magnitude, phase and zplane

    input:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    fs...sampling frequency in Hz
    output:
    bode plot as new figure
    """
    if fig is None:
        fig = plt.figure(figsize=(16, 9))

    p = np.roots(a)
    z = np.roots(b)
    W, Hd = sig.freqz(b, a, N)
    s, Ha = sig.freqs(B, A, fs*W)
    if Hd[0] == 0:
        Hd[0] = 1e-15  # avoid zero at DC for plotting dB
    if Ha[0] == 0:
        Ha[0] = 1e-15
    f = fs*W / (2*np.pi)

    gs = fig.add_gridspec(2, 2)
    # magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(f,
             20*np.log10(np.abs(Ha)),
             "C0",
             label=r'$|H(\omega)|$ continuous-time',
             linewidth=3)
    ax1.plot(f,
             20*np.log10(np.abs(Hd)),
             "C1",
             label=(r'$|H(\Omega)|$ discrete-time, fs=%5.f Hz' %fs),
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
    ax1.legend(loc="lower left")
    ax1.grid(True, which="both", axis="both",
             linestyle="-", linewidth=0.5, color=(0.8, 0.8, 0.8))

    # phase
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(f,
             (np.angle(Ha)*180/np.pi),
             "C0",
             label=r'$\mathrm{angle}(H(\omega))$ continuous-time',
             linewidth=3)
    ax2.plot(f,
             (np.angle(Hd)*180/np.pi),
             'C1',
             label=(r'$\mathrm{angle}(H(\Omega))$ discrete-time, fs=%5.f Hz' %fs),
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
    ax2.legend(loc="lower left")
    ax2.grid(True, which="both", axis="both",
             linestyle="-", linewidth=0.5, color=(0.8, 0.8, 0.8))

    # zplane
    ax3 = fig.add_subplot(gs[:, 1])
    zplane_plot(ax3, z, p)
