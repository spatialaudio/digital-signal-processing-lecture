"""Routines to create filter coefficients and plot filters.

for 1st / 2nd order audio biquads

URL = ('https://github.com/spatialaudio/'
       'digital-signal-processing-lecture/tree/master/'
       'filter_design')
"""
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle
import numpy as np
import scipy.signal as signal


def bilinear_biquad(B, A, fs=48000):
    """Get the bilinear transform of a 2nd-order Laplace transform.

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


def bw_from_q(QBP):
    """Convert bandpass quality to bandwidth in octaves."""
    return 2/np.log(2) * np.arcsinh(1/(2*QBP))


def q_from_bw(BWoct):
    """Convert bandwidth in ocatves to bandpass quality."""
    return 1 / (2*np.sinh(np.log(2)/2*BWoct))


def prewarping_f(f, fs=48000):
    """Do the frequency prewarping.

    input:
    f...analog frequency in Hz to be prewarped
    fs...sampling frequency in Hz
    output:
    prewarped digital angular frequency in rad/s
    """
    return 2*fs*np.tan(np.pi*f/fs)


def prewarping_q(Q, fm, fs=48000, WarpType="cos"):
    """Do the quality prewarping.

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
    """Calc coeff for lowpass 1st order.

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
    wcpre = prewarping_f(fc, fs)
    B = 0, 0, 1
    A = 0, 1 / wc, 1
    Bp = 0, 0, 1
    Ap = 0, 1 / wcpre, 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_lp2nd(fc, fs, bi=1, ai=np.sqrt(2)):
    """Calc coeff for lowpass 2nd order.

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
    wcpre = prewarping_f(fc, fs)
    B = 0, 0, 1
    A = bi / (wc**2), ai / wc, 1
    Bp = 0, 0, 1
    Ap = bi / (wcpre**2), ai / wcpre, 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_hp1st(fc, fs):
    """Calc coeff for highpass 1st order.

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
    wcpre = prewarping_f(fc, fs)
    B = 0, 1 / wc, 0
    A = 0, 1 / wc, 1
    Bp = 0, 1 / wcpre, 0
    Ap = 0, 1 / wcpre, 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_hp2nd(fc, fs, bi=1, ai=np.sqrt(2)):
    """Calc coeff for highpass 2nd order.

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
    wcpre = prewarping_f(fc, fs)
    B = 1 / (wc**2), 0, 0
    A = 1 / (wc**2), ai / wc, bi
    Bp = 1 / (wcpre**2), 0, 0
    Ap = 1 / (wcpre**2), ai / wcpre, bi
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_bp2nd(fm, Q, fs, QWarpType="cos"):
    """Calc coeff for bandpass 2nd order.

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
    wmpre = prewarping_f(fm, fs)
    Qpre = prewarping_q(Q, fm, fs, QWarpType)
    B = 0, 1 / (Q*wm), 0
    A = 1 / (wm**2), 1 / (Q*wm), 1
    Bp = 0, 1 / (Qpre*wmpre), 0
    Ap = 1 / (wmpre**2), 1 / (Qpre*wmpre), 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_bs2nd(fm, Q, fs, QWarpType="cos"):
    """Calc coeff for bandstop 2nd order.

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
    wmpre = prewarping_f(fm, fs)
    Qpre = prewarping_q(Q, fm, fs, QWarpType)
    B = 1 / wm**2, 0, 1
    A = 1 / wm**2, 1 / (Q*wm), 1
    Bp = 1 / wmpre**2, 0, 1
    Ap = 1 / wmpre**2, 1 / (Qpre*wmpre), 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_ap1st(fc, fs, ai=1):
    """Calc coeff for allpass 1st order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    ai...filter characteristics coefficients, e.g. ai = 1
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wcpre = prewarping_f(fc, fs)
    B = 0, -ai / wc, 1
    A = 0, +ai / wc, 1
    Bp = 0, -ai / wcpre, 1
    Ap = 0, +ai / wcpre, 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_ap2nd(fc, fs, bi=1, ai=np.sqrt(2)):
    """Calc coeff for allpass 2nd order.

    input:
    fc...cut frequency in Hz
    fs...sampling frequency in Hz
    bi, ai...filter characteristics coefficients, e.g.
    bi = 1, ai = 1.4142
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wcpre = prewarping_f(fc, fs)
    B = bi / (wc**2), -ai / wc, 1
    A = bi / (wc**2), +ai / wc, 1
    Bp = bi / (wcpre**2), -ai / wcpre, 1
    Ap = bi / (wcpre**2), +ai / wcpre, 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_peq2nd(fm, G, Q, fs, PEQType="III", QWarpType="cos"):
    """Calc coeff for peak/dip equalizer (PEQ) 2nd order.

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
    wmpre = prewarping_f(fm, fs)
    g = 10**(G/20)
    Qpre = prewarping_q(Q, fm, fs, QWarpType)
    if PEQType == "I":  # aka constant-Q PEQ
        gamma = g
        delta = g
    elif PEQType == "II":  # aka symmetrical PEQ
        gamma = 1
        delta = g
    elif PEQType == 'III':  # aka one-half pad loss PEQ
        gamma = g**0.5
        delta = g**0.5
    else:
        gamma = unknown_PEQType
        delta = unknown_PEQType
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = 0, 0, 1  # flat EQ
        A = 0, 0, 1
        b = 1, 0, 0
        a = 1, 0, 0
        return np.asarray(B), np.asarray(A), np.asarray(b), np.asarray(a)
    elif G > 0:
        B = 1 / wm**2, delta / (Q*wm), 1
        A = 1 / wm**2, (delta/g) / (Q*wm), 1
        Bp = 1 / wmpre**2, delta / (Qpre*wmpre), 1
        Ap = 1 / wmpre**2, (delta/g) / (Qpre*wmpre), 1
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a
    else:
        B = 1 / wm**2, gamma / (Q*wm), 1
        A = 1 / wm**2, (gamma/g) / (Q*wm), 1
        Bp = 1 / wmpre**2, gamma / (Qpre*wmpre), 1
        Ap = 1 / wmpre**2, (gamma/g) / (Qpre*wmpre), 1
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a


def biquad_peq2nd_zoelzer(fm, G, Q, fs):
    """Calc coeff for peak/dip equalizer (PEQ) 2nd order.

    according to
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
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = 1, 0, 0  # flat EQ
        a = 1, 0, 0
    elif G > 0:
        tmp = 1 + K/Q + K**2
        b = (1 + V0/Q * K + K**2) / tmp,\
            2 * (K**2 - 1) / tmp,\
            (1 - V0/Q * K + K**2) / tmp
        a = 1,\
            2 * (K**2 - 1) / tmp,\
            (1 - K/Q + K**2) / tmp
    else:
        tmp = 1 + K / (V0*Q) + K**2
        b = (1 + K/Q + K**2) / tmp,\
            2 * (K**2 - 1) / tmp,\
            (1 - K/Q + K**2) / tmp
        a = 1,\
            2 * (K**2 - 1) / tmp,\
            (1 - K/(V0*Q) + K**2) / tmp
    return np.asarray(b), np.asarray(a)


def biquad_peq2nd_RBJ(fm, G, Q, fs):
    """Calc coeff for peak/dip equalizer (PEQ) 2nd order.

    according to
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
    Ksqrt = 10**(G/40)
    w0 = 2*np.pi*fm / fs
    BW = bw_from_q(Q)
    gamma = np.sinh(np.log(2)/2 * (BW*w0) / np.sin(w0))*np.sin(w0)
    tmp = 1 + gamma/Ksqrt
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = 1, 0, 0  # flat EQ
        a = 1, 0, 0
    else:
        b = (1 + gamma*Ksqrt) / tmp,\
            -2 * np.cos(w0) / tmp,\
            (1 - gamma*Ksqrt) / tmp
        a = 1,\
            -2 * np.cos(w0) / tmp,\
            (1 - gamma/Ksqrt) / tmp
    return np.asarray(b), np.asarray(a)


def biquad_lshv1st(fc, G, fs, ShvType="III"):
    """Calc coeff for lowshelving 1st order.

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
    wcpre = prewarping_f(fc, fs)
    g = 10**(G/20)
    if ShvType == "I":
        alpha = 1
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = 0, 0, 1  # flat EQ
        A = 0, 0, 1
        b = 1, 0, 0
        a = 1, 0, 0
        return np.asarray(B), np.asarray(A), np.asarray(b), np.asarray(a)
    elif G > 0:
        B = 0, 1 / wc, g * alpha**-2
        A = 0, 1 / wc, alpha**-2
        Bp = 0, 1 / wcpre, g * alpha**-2
        Ap = 0, 1 / wcpre, alpha**-2
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a
    else:
        B = 0, 1 / wc, alpha**2
        A = 0, 1 / wc, g**-1 * alpha**2
        Bp = 0, 1 / wcpre, alpha**2
        Ap = 0, 1 / wcpre, g**-1 * alpha**2
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a


def biquad_lshv2nd(fc, G, fs,
                   ShvType="III", Qz=1/np.sqrt(2), Qp=1/np.sqrt(2)):
    """Calc coeff for lowshelving 2nd order.

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    ShvType..."I", "II", "III"
    Qz...zero Quality, e.g. Qz = 1/np.sqrt(2) for Butterworth quality
    Qp...pole quality, e.g. Qp = 1/np.sqrt(2) for Butterworth quality
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    g = 10**(G/20)
    wc = 2*np.pi*fc
    wcpre = prewarping_f(fc, fs)
    if ShvType == "I":
        alpha = 1
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = 0, 0, 1  # flat EQ
        A = 0, 0, 1
        b = 1, 0, 0
        a = 1, 0, 0
        return np.asarray(B), np.asarray(A), np.asarray(b), np.asarray(a)
    elif G > 0:
        B = 1 / wc**2, g**0.5 * alpha**-1 / (Qz*wc), g * alpha**-2
        A = 1 / wc**2, alpha**-1 / (Qp*wc), alpha**-2
        Bp = 1 / wcpre**2, g**0.5 * alpha**-1 / (Qz*wcpre), g * alpha**-2
        Ap = 1 / wcpre**2, alpha**-1 / (Qp*wcpre), alpha**-2
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a
    else:
        B = 1 / wc**2, alpha / (Qz*wc), alpha**2
        A = 1 / wc**2, g**-0.5 * alpha / (Qp*wc), g**-1 * alpha**2
        Bp = 1 / wcpre**2, alpha / (Qz*wcpre), alpha**2
        Ap = 1 / wcpre**2, g**-0.5 * alpha / (Qp*wcpre), g**-1 * alpha**2
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a


def biquad_lshv2nd_Zoelzer(fc, G, fs):
    """Calc coeff for highshelving 2nd order.

    according to
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
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = 1, 0, 0  # flat EQ
        a = 1, 0, 0
    elif G > 0:
        tmp = 1 + np.sqrt(2)*K + K**2
        b = (1 + np.sqrt(2*V0)*K + V0*K**2) / tmp, 2 * (V0 * K**2 - 1) / tmp,\
            (1 - np.sqrt(2*V0)*K + (V0 * K**2)) / tmp
        a = 1, 2 * (K**2 - 1) / tmp, (1 - np.sqrt(2)*K + K**2) / tmp
    else:
        tmp = V0 + np.sqrt(2*V0)*K + K**2
        b = V0 * (1 + np.sqrt(2)*K + K**2) / tmp, 2*V0 * (K**2 - 1) / tmp,\
            V0 * (1 - np.sqrt(2)*K + K**2) / tmp
        a = 1, 2 * (K**2 - V0) / tmp, (V0 - np.sqrt(2*V0)*K + K**2) / tmp
    return np.asarray(b), np.asarray(a)


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
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    A = 10**(G/40)
    w0 = 2*np.pi*fc / fs
    alpha = np.sin(w0)/2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = 1, 0, 0  # flat EQ
        a = 1, 0, 0
    else:
        b = A * ((A + 1) - (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha),\
            2*A * ((A - 1) - (A + 1)*np.cos(w0)),\
            A * ((A + 1) - (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
        a = (A + 1) + (A - 1)*np.cos(w0) + 2*np.sqrt(A)*alpha,\
            -2 * ((A - 1) + (A + 1)*np.cos(w0)),\
            (A + 1) + (A - 1)*np.cos(w0) - 2*np.sqrt(A)*alpha
        b = b / a[0]  # relies on a[0]
        a = a / a[0]
    return np.asarray(b), np.asarray(a)


def biquad_hshv1st(fc, G, fs, ShvType="III"):
    """Calc coeff for highshelving 1st order.

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
    wcpre = prewarping_f(fc, fs)
    g = 10**(G/20)
    if ShvType == "I":
        alpha = 1
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = 0, 0, 1  # flat EQ
        A = 0, 0, 1
        b = 1, 0, 0
        a = 1, 0, 0
        return np.asarray(B), np.asarray(A), np.asarray(b), np.asarray(a)
    if G > 0:
        B = 0, g * alpha**-2 / wc, 1
        A = 0, alpha**-2 / wc, 1
        Bp = 0, g * alpha**-2 / wcpre, 1
        Ap = 0, alpha**-2 / wcpre, 1
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a
    else:
        B = 0, alpha**2 / wc, 1
        A = 0, g**-1 * alpha**2 / wc, 1
        Bp = 0, alpha**2 / wcpre, 1
        Ap = 0, g**-1 * alpha**2 / wcpre, 1
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a


def biquad_hshv2nd(fc, G, fs,
                   ShvType="III", Qz=1/np.sqrt(2), Qp=1/np.sqrt(2)):
    """Calc coeff for highshelving 2nd order.

    input:
    fc...cut frequency in Hz
    G...gain or attenuation in dB
    fs...sampling frequency in Hz
    ShvType..."I", "II", "III"
    Qz...zero Quality, e.g. Qz = 1/np.sqrt(2) for Butterworth quality
    Qp...pole quality, e.g. Qp = 1/np.sqrt(2) for Butterworth quality
    output:
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    wc = 2*np.pi*fc
    wcpre = prewarping_f(fc, fs)
    g = 10**(G/20)
    if ShvType == "I":
        alpha = 1
    elif ShvType == "II":
        alpha = g**0.5
    elif ShvType == "III":  # one-half pad loss characteristics
        alpha = g**0.25
    else:
        alpha = unknown_ShvType
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        B = 0, 0, 1  # flat EQ
        A = 0, 0, 1
        b = 1, 0, 0
        a = 1, 0, 0
        return np.asarray(B), np.asarray(A), np.asarray(b), np.asarray(a)
    if G > 0:
        B = g * alpha**-2 / wc**2, g**0.5 * alpha**-1 / (Qz*wc), 1
        A = alpha**-2 / wc**2, alpha**-1 / (Qp*wc), 1
        Bp = g * alpha**-2 / wcpre**2, g**0.5 * alpha**-1 / (Qz*wcpre), 1
        Ap = alpha**-2 / wcpre**2, alpha**-1 / (Qp*wcpre), 1
        b, a = bilinear_biquad(Bp, Ap, fs)
        return np.asarray(B), np.asarray(A), b, a
    else:
        B = alpha**2 / wc**2, alpha / (Qz*wc), 1
        A = g**-1 * alpha**2 / wc**2, g**-0.5 * alpha / (Qp*wc), 1
        Bp = alpha**2 / wcpre**2, alpha / (Qz*wcpre), 1
        Ap = g**-1 * alpha**2 / wcpre**2, g**-0.5 * alpha/(Qp*wcpre), 1
    b, a = bilinear_biquad(Bp, Ap, fs)
    return np.asarray(B), np.asarray(A), b, a


def biquad_hshv2nd_Zoelzer(fc, G, fs):
    """Calc coeff for highshelving 2nd order.

    according to
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
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = 1, 0, 0  # flat EQ
        a = 1, 0, 0
    elif G > 0:
        tmp = 1 + np.sqrt(2)*K + K**2
        b = (V0 + np.sqrt(2*V0)*K + K**2) / tmp, 2 * (K**2 - V0) / tmp,\
            (V0 - np.sqrt(2*V0)*K + K**2) / tmp
        a = 1, 2 * (K**2 - 1) / tmp, (1 - np.sqrt(2)*K + K**2) / tmp
    else:
        tmp = 1 + np.sqrt(2*V0)*K + (V0 * K**2)
        b = V0 * (1 + np.sqrt(2)*K + K**2) / tmp, 2*V0 * (K**2 - 1) / tmp,\
            V0 * (1 - np.sqrt(2.)*K + K**2) / tmp
        a = 1, 2 * (V0 * K**2 - 1) / tmp,\
            (1 - np.sqrt(2*V0)*K + (V0 * K**2)) / tmp
    return np.asarray(b), np.asarray(a)


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
    B...numerator cofficients Laplace transfer function
    A...denominator cofficients Laplace transfer function
    b...numerator coefficients z-transfer function
    a...denominator cofficients z-transfer function
    """
    A = 10**(G/40)
    w0 = 2*np.pi*fc / fs
    alpha = np.sin(w0)/2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    if np.isclose(G, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
        b = 1, 0, 0  # flat EQ
        a = 1, 0, 0
    else:
        b = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha),\
            -2 * A * ((A - 1) + (A + 1) * np.cos(w0)),\
            A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha,\
            2 * ((A - 1) - (A + 1) * np.cos(w0)),\
            (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        b = b / a[0]  # relies on a[0]
        a = a / a[0]
    return np.asarray(b), np.asarray(a)


def zplane_plot(ax, z, p):
    """Realize a zplane plot.

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
            markersize=8, markeredgewidth=2)
    ax.plot(np.real(p), np.imag(p),
            "x", label="poles",
            color="C3", fillstyle="none",
            markersize=8, markeredgewidth=2)
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


def bode_plot(B, A, b, a, fs, figsize=(10, 6.25), fig=None, N=2**12):
    """Realize a bode plot containing magnitude, phase and zplane.

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
        fig = plt.figure()
    fig.set_size_inches(figsize)    
    p = np.roots(a)
    z = np.roots(b)
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
    zplane_plot(ax3, z, p)

    print("B =", B)
    print("A =", A)
    print("b =", b)
    print("a =", a)


def magnitude_plot_overlay(x, y, title, legend, figsize=(6, 3.75), fig=None):
    """Realize a bode plot containing magnitude for overlay."""
    if fig is None:
        fig = plt.figure()
    fig.set_size_inches(figsize)     
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
