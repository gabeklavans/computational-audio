"""
Note: these functions are designed more to be clear and/or concise rather than optimized for speed
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def parabolicInterpolationPeak(xs, ys):
    """
    Find the interpolated peak between 3 points using parabolic interpolation

    ~~Args~~

    xs: list size 3
        list of function inputs
        
    ys: list size 3
        list of function outputs
    """
    [x1, x2, x3] = xs
    [y1, y2, y3] = ys
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    f = interp1d(xs, ys, kind="quadratic")
    x = -B / (2 * A)
    return (x, f(x))


def autocorrelation(
    sig,
    sample_rate,
    start=0,
    window_size=4410,
    relative_threshold=0.75,
    interpolate=True,
):
    """
    Autocorrelation f0 estimation

    ~~Args~~

    sig: array-like
        all the signal amplitude values with respect to sample-rate based time

    start: int (default = 0)
        the index in the signal to start the analysis at

    window_size: int (default = 4410)
        size of the window cut from the signal to be correlated NOTE: default is based on a sample_rate=44100

    relative_threshold: float (default = 0.75)
        threshold for considering a correlation value acceptable, relative to the maximum correlation value; used for finding significant peaks

    interpolation: boolean (default = True)
        whether to interpolate the position in time where the first ac peak is found
    """

    # normalize to prevent overflow
    max_amp = np.max(sig)
    sig = np.array([sig_val / max_amp for sig_val in sig])
    # get stationary lagged window
    X_lagged = sig[start : start + window_size]

    # correlations are index by lag k
    correlations = np.array(
        [
            X_lagged @ sig[(start + offset) : (start + offset + window_size)]
            for offset in range(window_size)
        ]
    )
    # normalize for relative threshold
    correlations = correlations / np.max(correlations)

    peaks, _ = find_peaks(correlations)
    filtered_peaks = peaks[correlations[peaks] > relative_threshold]

    if interpolate:
        peak = filtered_peaks[0] if len(filtered_peaks) > 0 else np.argmax(correlations)
        # perform manual interpolation
        xs = [peak - 1, peak, peak + 1]
        ys = [
            correlations[peak - 1],
            correlations[peak],
            correlations[peak + 1],
        ]
        interpolated_peak, _ = parabolicInterpolationPeak(xs, ys)

        return sample_rate / interpolated_peak
    else:
        # estimate using first peak above threshold
        # else, estimate using the global max
        return (
            sample_rate / filtered_peaks[0]
            if len(filtered_peaks) > 0
            else sample_rate / np.argmax(correlations)
        )


def yin(
    sig,
    sample_rate,
    t=0,
    window_size=4410,
    threshold=0.1,
    interpolate=True,
    display=False,
):
    """
    YIN f0 estimation

    Author: Wayne Snyder

    Date: Jul 30 2020

    Description: This is an implementation of the Yin Algorithm, from http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf
    """

    # normalize X to prevent overflow when multiplying
    max_amp = np.max(sig)
    sig = np.array([x / max_amp for x in sig])

    W = window_size // 2  # can only calculate lag times up to half of window size

    X_lagged = sig[t : (t + W)]  # stationary window of width W

    # Calculate d_t(tau),  p.1919

    d_t = np.array(
        [np.sum((X_lagged - sig[(tau + t) : (tau + t + W)]) ** 2) for tau in range(W)]
    )

    # Calculate the cumulative mean normalized difference function d1_t(tau) = d'_t(tau)

    #  d1_t(tau) = 1 for tau = 0
    #            = d_t(tau) / mean(d_t(1), ..., d_t(tau))  otherwise
    #              = tau*d_t(tau)/(d_t(1)+ ... +d_t(tau))

    # Cumulative sum of [x0,x1,x2, .... ] is [x0, x0+x1, x0+x1+x2, ... ]
    # so this will calculate the denominator above:

    cumm_d_t = np.cumsum(d_t)

    d1_t = np.array(
        [1 if (tau == 0) else tau * d_t[tau] / cumm_d_t[tau] for tau in range(W)]
    )

    # Now calculate minima in d1_t and find first one that is below
    # a fixed threshold; if none is below threshold, then choose overall global minimum

    # note that you can't choose tau as last slot (len(d1_t)-1) because such a tau =
    # can never be a "valley" (it doesn't have a right neighbor to be less than).
    # Similarly, tau can not be < 3 because tau = 2 corresponds to period 2, and
    # frequency 44100/2 = 22050 = Nyquist Limit!

    tau = None

    # find valleys below the threshold
    for t1 in range(3, len(d1_t) - 1):
        if d1_t[t1] < threshold and d1_t[t1 - 1] > d1_t[t1] and d1_t[t1] < d1_t[t1 + 1]:
            tau = t1
            break

    if tau == None:
        print("Warning: Using global minimum!")
        # index of first global minimum
        # but not the very last slot (cf. len(d1_t)-1 above), so find min in d1_t[:-1]
        tau = (np.where(d1_t[:-1] == np.amin(d1_t[:-1])))[0][0]

    if interpolate:
        xs = [tau - 1, tau, tau + 1]
        ys = [d1_t[tau - 1], d1_t[tau], d1_t[tau + 1]]

        tau, _ = parabolicInterpolationPeak(xs, ys)

    if display:

        plt.figure(figsize=(10, 5))
        plt.title("Signal Window")
        plt.plot(X_lagged)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.title("Difference Function")
        plt.plot(d_t)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.title("Normalized Difference Function")
        plt.plot(d1_t)
        plt.show()

    # return (sample_rate / tau, d_t, d1_t)
    return sample_rate / tau


def cepstrum(
    sig,
    sample_rate,
    start=0,
    window_size=4410,
    f_min=82,
    f_max=640,
    interpolate=True,
    display=False,
):
    """
    Cepstrum f0 estimation

    Source of method: http://flothesof.github.io/cepstrum-pitch-tracking.html#What-is-a-cepstrum?

    ~~Args~~

    sig: array-like
        the signal

    sample_rate: int
        sample rate of recorded signal

    start: int (default = 0)
        starting signal index of the window

    window_size: int (default = 4410)
        size of analysis window

    f_min: int (default = 82)
        minimum detectable frequency

    f_max: int (default = 640)
        maximum detectable frequency (too high can introduce octave errors)

    interpolate: boolean (default = True)
        interpolate the maximum peak in the cepstrum

    display: boolean (default = False)
        display figures relating to the calculations
    """
    sig = np.array(sig[start : start + window_size])
    sig = np.hanning(len(sig)) * sig  # window func
    spectrum = np.fft.rfft(sig)
    spectrum = np.abs(spectrum)
    log_spectrum = np.log(spectrum)

    freq_axis = np.fft.rfftfreq(len(sig), d=1 / sample_rate)

    cepstrum = np.fft.rfft(log_spectrum)
    cepstrum = np.abs(cepstrum)

    dt = freq_axis[1] - freq_axis[0]
    quefrency_axis = np.fft.rfftfreq(len(spectrum), d=dt)

    if display:
        plt.title("(Shortened) Cepstrum")
        plt.plot(quefrency_axis[:100], cepstrum[:100])
        plt.show()

    valid_quefrencies = (quefrency_axis > 1 / f_max) & (quefrency_axis < 1 / f_min)

    max_quefrency_index = np.argmax(cepstrum[valid_quefrencies])

    if display:
        plt.title("Argmax within valid range")
        plt.plot(quefrency_axis[:100], cepstrum[:100])
        plt.scatter(
            quefrency_axis[valid_quefrencies][max_quefrency_index],
            cepstrum[valid_quefrencies][max_quefrency_index],
            color="r",
        )
        plt.show()

    if interpolate:
        quefrency_axis_valid = quefrency_axis[valid_quefrencies]
        cepstrum_valid = cepstrum[valid_quefrencies]
        quefrency_interped = interp1d(
            np.arange(len(quefrency_axis_valid)), quefrency_axis_valid
        )
        # set up interp range
        rang = [
            max_quefrency_index - 1,
            max_quefrency_index,
            max_quefrency_index + 1,
        ]
        px, _ = parabolicInterpolationPeak(rang, [cepstrum_valid[i] for i in rang])

        f0_guess = 1 / quefrency_interped(px)

        if display:
            plt.title("Interpolated max")
            plt.plot(quefrency_axis[:100], cepstrum[:100])
            plt.scatter(
                quefrency_interped(px), cepstrum_valid[max_quefrency_index], color="r"
            )
            plt.show()

        return f0_guess
    else:
        return 1 / quefrency_axis[valid_quefrencies][max_quefrency_index]


def landmark_points(
    signal,
    sample_rate,
    start=0,
    window_size=800,
    distance_thresh=5,
    similarity_thresh=0.75,
    display=False,
    graph_cutoff=800,
):
    """
    Landmark points f0 estimation

    Implementation of Waveform Periodicity Determination using Landmark Points algorithm
    described in paper by the same name by Copper & Ng, published University of Leeds https://www.jstor.org/stable/3680825?seq=1

    ~~Args~~

    sig: array-like
        the signal

    sample_rate: int
        sample rate of recorded signal

    start: int (default = 0)
        starting signal index of the window

    window_size: int (default = 4410)
        size of analysis window

    distance_thresh: int (default = 5)
        maximum difference in segment lengths; optimal default from pg. 72

    similarity_thresh: int (default = 0.75)
        minimum similarity between largest segment and comparison segment; optimal default from pg. 72

    display: boolean (default = False)
        display figures relating to the calculations

    graph_cutoff: int (default = 800)
        number of samples to display in graph (the graph can get kinda busy)
    """
    ### get windowed signal
    # default window size suggested on pg. 72
    signal = signal[start : start + window_size]

    ### get positive-going zero-crossings, or segment delimeters
    positive_going_zero_crossings = (np.diff(np.sign(signal + 0.000001)) > 0).nonzero()[
        0
    ] + 1
    # Hacky way to treat 0's as positive is add a really small number.
    # Reason for doing this is if the the signal happens to sample exactly at 0, it will double count it
    # as a positive-going zero crossing, when we just want the transition from negative to 0.

    ### get the locations of the sub-segments
    sub_seg_points = []

    for i in range(0, len(positive_going_zero_crossings) - 1):
        seg_start = positive_going_zero_crossings[i]
        seg_end = positive_going_zero_crossings[i + 1]

        # get delta size for each subsegment in the segment
        d_sub_seg = (seg_end - seg_start) // 8

        # ignore segments with less that 8 samples
        if d_sub_seg > 0:
            for j in range(8):
                sub_seg_points.append(seg_start + (d_sub_seg * j))

    sub_seg_points = np.array(sub_seg_points)

    ### get the 6 landmarks for each segment, store them per-segment
    seg_landmarks = []

    for seg in range(len(sub_seg_points) // 8):
        landmarks = []
        for i in range(0, 3):
            landmarks.append(sub_seg_points[(i + 1) + (seg * 8)])
        for i in range(4, 7):
            landmarks.append(sub_seg_points[(i + 1) + (seg * 8)])
        seg_landmarks.append(landmarks)

    seg_landmarks = np.array(seg_landmarks)

    if display:
        plt_range = graph_cutoff

        plt.figure(figsize=[15, 8])
        plt.plot(signal[:plt_range])
        plt.axhline(linewidth=1, linestyle="-", c="black")
        for x in sub_seg_points[sub_seg_points < plt_range]:
            plt.axvline(x=x, linewidth=1, linestyle=":", c="black", alpha=1)
        for x in positive_going_zero_crossings[
            positive_going_zero_crossings < plt_range
        ]:
            plt.axvline(x=x, linewidth=2, linestyle="--", c="black")
        plt.scatter(
            seg_landmarks.flatten()[seg_landmarks.flatten() < plt_range],
            signal[seg_landmarks.flatten()][seg_landmarks.flatten() < plt_range],
            marker="s",
            facecolors="none",
            edgecolors="r",
        )

    ### finally, calculate a guess
    seg_lengths = np.diff(positive_going_zero_crossings)
    # get index of longest seg
    # every other seg will be compared to this one
    longest_seg_idx = np.argmax(seg_lengths)

    for i in range(len(seg_landmarks)):
        if i != longest_seg_idx:
            longest_seg = seg_landmarks[longest_seg_idx]
            seg = seg_landmarks[i]
            similarity_ratio = np.dot(longest_seg, seg)  # similarity metric, pg. 72

            # check if the seg is similar enough
            # Optimal distance and similarity thresholds from pg. 73 set as default
            if (
                abs(seg_lengths[i] - seg_lengths[longest_seg_idx]) < distance_thresh
            ) and (similarity_ratio > similarity_thresh):
                dist_between_segs = 0
                start = longest_seg_idx if longest_seg_idx < i else i
                end = longest_seg_idx if longest_seg_idx > i else i
                for j in range(start, end):
                    dist_between_segs += seg_lengths[j]
                if dist_between_segs == 0:
                    # if the segs are adjacent
                    sample_rate / seg_lengths[longest_seg_idx]
                else:
                    # formula for getting freq est, pg. 73
                    return sample_rate / (dist_between_segs)

    # if can't be calculated, just use the largest segment
    return sample_rate / seg_lengths[longest_seg_idx]

