from scipy.stats import truncnorm
import numpy as np
import warnings


# - Functions for constructing the ecg segments


def swave(t, t0, t1, y0, y1):
    """ Return the value of a sigmoid that takes values y0 for t<t0,
    y1 for t>t1, 0.5*(y0+y1) for t=0.5*(t0+t1).
    Can be used to smoothly change where a vhull or vQRS begins or
    ends.
        t : point where signoid is evaluated - can be an array
        t0 : beginning of sigmoid
        t1 : end of sigmoid
        y0 : value at t0
        y1 : value at t1
    """
    return (
        0.5
        * (y0 + y1 + (y0 - y1) * np.cos(np.pi * (t - t0) / (t1 - t0)))
        * (t >= t0)
        * (t <= t1)
    )


def vhull(tLength, fYa=0, fYb=0, fSkew=0, fNotch=0):
    """Use the cosine function to produce a smoothly changing sequence
    of length tLength, starting at fYa, having an extremum in 1 and
    finishing at fYb. The derivative at 0 and tLength is 0. A notch
    can be added at the extremum to produce P mitrales.
    Can be used for P- and T-waves
        tLength : number of points
        fYa : first value
        fYb : last value
        fSkew : Changes symmetry of the resulting curve. If fSkew is
                0, the extremum is at tLength/2. If it is positive
                (negative), the extremum is shifted to the right (left).
        fNotch : Size of notch to be added.
    """
    fP = 1 + abs(fSkew)
    if fSkew >= 0:
        t0 = 0
        t1 = tLength
    else:
        t0 = tLength
        t1 = 0
    vt = np.arange(tLength)
    # Bump, where first and last points are 0, extremum is 1, shifted
    # away from center if fP != 1
    vBump = 0.5 * (1. - np.cos(2. * np.pi * np.abs((vt - t0) / (t1 - t0)) ** fP))
    # Location of extremum
    tExtr = int(0.5 ** (1. / fP) * (t1 - t0) + t0)
    # Add notch
    if fNotch > 1e-5:
        nHalfNotchWidth = int(0.25 * tLength)
        vNotch = (
            -0.5
            * fNotch
            * (1. - np.cos(np.pi / nHalfNotchWidth * vt[: 2 * nHalfNotchWidth]))
        )
        vBump[tExtr - nHalfNotchWidth : tExtr + nHalfNotchWidth] += vNotch
    # Change the values of first and last points, such that derivative there
    # is still 0 and curve remains smooth
    vElevL = swave(vt, 0, tExtr, fYa, 0)
    vElevR = swave(vt, tExtr, tLength, 0, fYb)
    return vBump + vElevL + vElevR


def vQRS(tLength, fSQ, fYa=0, fYb=0):
    """Produce QRS complex of length tLength, starting at fYa, ending
    with fYb. The curve is smooth, derivatives are 0 in beginning and
    end. The large peak has height 1.
        tLength : number of points
        fSQ : ratio between the two minima befor and after large spike
        fYa : first value
        fYb : last value
    """
    vt = np.arange(tLength)
    sig = np.zeros(tLength)
    t0 = int(tLength / 4)
    t1 = int(3 * tLength / 4)
    # First minimum
    sig[:t0] = -np.sin(4. / tLength * np.pi * vt[:t0]) * (vt[:t0] / tLength) ** 4
    # Second minimum
    sig[t1:] = (
        -np.sin(4. / tLength * np.pi * (tLength - vt[t1:]))
        * ((tLength - vt[t1:]) / tLength) ** 4
    )
    # Large spike
    sig[t0:t1] = 1. / 128. * np.sin(2. * np.pi / tLength * (vt[t0:t1] - tLength / 4.))
    # Normalize so that peak is 1.
    sig /= np.amax(np.abs(sig))
    # Change the values of first and last points, such that derivative there
    # is still 0 and curve remains smooth
    sig += swave(vt, 0, 0.5 * tLength, fYa, 0) + swave(
        vt, 0.5 * tLength, tLength, 0, fYb
    )
    return sig


# @profile
def labeled_signal(
    n : int,
    fHeartRate : float,
    lRepertoire: str = "all",
    dProbs: dict = None,
    fStdNoise: float = 0,
    **kwargs):
    """Create 2D array of n ecg segments and rhythms and their labels.
    Both single normal or anomal segments as well as complete normal
    or anomal ecg rhythms can be included. Complete rhythms are
    labelled segment-wise, so that the only difference to producing
    single ecg segments is that the segments are in order. The third
    row of the returned array indicates which segments belong to a
    "'complete_normal'"-rhythm.
        n : number of segments
        fHeartRate : frequency (in 1/timepoints) of complete sinus rhythm
        lRepertoire : which kinds of segments to produce - either list of
                      segment types or 'all'
        dProbs : dictionary with probabilities for given elements
        fStdNoise :  float Standard deviation of Gaussian noise added to
                                  ECG signal
        kwargs : dict to modify default parameters that are otherwise
                 defined in dParameters
    """
    n = int(n)
    dParameters = {
        "t1": 0.165,
        "t2": 0.25,
        "t3": 0.375,
        "t4": 0.49,
        "t5": 0.575,
        "t6": 0.775,
        "P": 0.5,
        "P_skew": 0.5,
        "P_mitrale": 0.5,
        "R": 5,
        "SQ": 3,
        "T": 1,
        "T_skew": 0.5,
        "ST_elev": 0.9,
        "ST_depr": -0.8,
        "stdev": 0.1,
    }
    dnLabels = {
        "flat": 0,
        "P": 1,
        "QRS": 2,
        "T": 3,
        "P_inv": 4,
        "no_P": 5,
        "no_QRS": 6,
        "T_inv": 7,
        "ST_elev": 8,
        "ST_depr": 9,
        "tachycardia": 10,
        "bradycardia": 11,
        "P_mitrale": 12,
    }
    # Second-stage labels
    dnLabels1 = {
        "complete_normal": 0,
        "complete_noP": 1,
        "complete_Pinv": 2,
        "complete_noQRS": 3,
        "complete_Tinv": 4,
        "complete_STelev": 5,
        "complete_STdepr": 6,
        "complete_tach": 7,
        "complete_brad": 8,
        "other": 9,
    }
    # Mean durations for each type of rhythm or segment
    dMeanDurs = {
        "P": (dParameters["t2"] - dParameters["t1"]) / fHeartRate,
        "QRS": (dParameters["t4"] - dParameters["t3"]) / fHeartRate,
        "T": (dParameters["t6"] - dParameters["t5"]) / fHeartRate,
        "flat": (1 - dParameters["t6"]) / fHeartRate,
        "complete_normal": 1. / fHeartRate,
        "complete_brad": 2. / fHeartRate,
        "complete_tach": 0.3 / fHeartRate,
    }
    # For each type of rhythm or segment, the number of random variables needed
    # This way RVs can be drawn at once, which is more efficient
    dnRVs = {
        "P": 3,  # duration, skew, height
        "P_inv": 3,  # duration, skew, height
        "P_mitrale": 3,  # duration, skew, height
        "QRS": 3,  # duration, QS-ratio, height
        "T": 3,  # duration, skew, height
        "T_inv": 3,  # duration, skew, height
        "flat": 1,  # duration
        "complete_normal": 13,  # 7 durations, rest for P,QRS,T
        "complete_noP": 13,  # 7 durations, rest for P,QRS,T
        "complete_Pinv": 13,  # 7 durations, rest for P,QRS,T
        "complete_noQRS": 13,  # 7 durations, rest for P,QRS,T
        "complete_Tinv": 13,  # 7 durations, rest for P,QRS,T
        "complete_STelev": 14,  # 7 durations, rest for P,QRS,T
        "complete_STdepr": 14,  # 7 durations, rest for P,QRS,T
        "complete_tach": 13,  # 7 durations, rest for P,QRS,T
        "complete_brad": 13,
    }  # 7 durations, rest for P,QRS,T

    # @profile
    def rhythm_timings(n, vRVs, tMeanDur, dParams):
        """Return n-vector of randomly drawn durations for n ecg rhythms
        with mean tMeanDur and nx7-matrix of corresponding segment timings.
            n : number of rhythms that durations and timings are drawn for.
            vRVs : random variables used to determine durations and timings
            tMeanDur : mean duration of rhythms
            dParams : dict with entries t1,..,t6 that determine timings
                      within rhythms.
        """
        vDurations = np.array(tMeanDur * (1 + dParams["stdev"] * vRVs[:n]), int)
        # Mean timings within single rhythm of length 1
        vMeanTimings = np.r_[0, [dParams["t{}".format(t)] for t in range(1, 7)]]
        # Drawing intervals instead of points in time avoids overlapping of segments
        vMeanIntervals = np.diff(vMeanTimings)
        # nx7 matrix of intervals from 0 to t1, t1 to t2,..,t6 to end
        mIntervals = np.array(
            [
                m * (dParams["stdev"] * vRVs[i * n : (i + 1) * n] + 1)
                for i, m in enumerate(vMeanIntervals)
            ]
        ).T
        mTimings = np.cumsum(mIntervals, axis=1)
        mTimings *= vDurations.reshape(-1, 1)
        return vDurations, np.asarray(mTimings, dtype=int)

    # @profile
    def append_rhythms(lSegments, zParameters, sAnomaly=None):
        """Create ecg rhythms and append them to lSegments. Shapes are
        determined by zParameters, labels are according to sAnomaly.
            lSegments : list to which new rhythms are appended
            zParameters : zip of arrays of parameters needed to determine
                          the shape of each rhythm: durations, timings,
                          P, R, SQ, T, Pskew, Tskew, STelevation

            vNewRhythm, axis 0:
                0: Actual ECG signal
                1: Segment label
                2: Whole rhythm label
                3: Segment label, only at first point of segment
                4: Segment label, only at final point of segment
        """
        for d, vi, p, r, q, t, ps, ts, ste in zParameters:
            vNewRhythm = np.zeros((5, d))
            vNewRhythm[3, :] = np.nan
            vNewRhythm[4, :] = np.nan
            # Indicate if normal rhythm in third row of array
            vNewRhythm[2, :] = dnLabels1[sAnomaly]
            # Initial flat segment
            vNewRhythm[1, : vi[0]] = dnLabels["flat"]
            vNewRhythm[3, 0] = dnLabels["flat"]
            vNewRhythm[4, vi[0] - 1] = dnLabels["flat"]
            # P-wave
            vNewRhythm[0, vi[0] : vi[1]] = p * vhull(vi[1] - vi[0], fSkew=ps)
            if sAnomaly == "complete_Pinv":
                vNewRhythm[1, vi[0] : vi[1]] = dnLabels["P_inv"]
                vNewRhythm[3, vi[0]] = dnLabels["P_inv"]
                vNewRhythm[4, vi[1] - 1] = dnLabels["P_inv"]
            elif sAnomaly == "complete_noP":
                vNewRhythm[1, vi[0] : vi[1]] = dnLabels["no_P"]
                vNewRhythm[3, vi[0]] = dnLabels["no_P"]
                vNewRhythm[4, vi[1] - 1] = dnLabels["no_P"]
            else:
                vNewRhythm[1, vi[0] : vi[1]] = dnLabels["P"]
                vNewRhythm[3, vi[0]] = dnLabels["P"]
                vNewRhythm[4, vi[1] - 1] = dnLabels["P"]
            # PQ segment
            vNewRhythm[1, vi[1] : vi[2]] = dnLabels["flat"]
            vNewRhythm[3, vi[2]] = dnLabels["flat"]
            vNewRhythm[4, vi[3] - 1] = dnLabels["flat"]
            # QRS complex
            vNewRhythm[0, vi[2] : vi[3]] = r * vQRS(
                vi[3] - vi[2], q, fYb=ste / max(abs(r), 0.01)
            )
            if sAnomaly == "complete_noQRS":
                vNewRhythm[1, vi[2] : vi[3]] = dnLabels["no_QRS"]
                vNewRhythm[3, vi[2]] = dnLabels["no_QRS"]
                vNewRhythm[4, vi[3] - 1] = dnLabels["no_QRS"]
            else:
                vNewRhythm[1, vi[2] : vi[3]] = dnLabels["QRS"]
                vNewRhythm[3, vi[2]] = dnLabels["QRS"]
                vNewRhythm[4, vi[3] - 1] = dnLabels["QRS"]
            # ST segment
            if sAnomaly == "complete_STelev":
                vNewRhythm[0, vi[3] : vi[4]] = ste
                vNewRhythm[1, vi[3] : vi[4]] = dnLabels["ST_elev"]
                vNewRhythm[3, vi[3]] = dnLabels["ST_elev"]
                vNewRhythm[4, vi[4] - 1] = dnLabels["ST_elev"]
            elif sAnomaly == "complete_STdepr":
                vNewRhythm[0, vi[3] : vi[4]] = ste
                vNewRhythm[1, vi[3] : vi[4]] = dnLabels["ST_depr"]
                vNewRhythm[3, vi[3]] = dnLabels["ST_depr"]
                vNewRhythm[4, vi[4] - 1] = dnLabels["ST_depr"]
            else:
                vNewRhythm[1, vi[3] : vi[4]] = dnLabels["flat"]
                vNewRhythm[3, vi[3]] = dnLabels["flat"]
                vNewRhythm[4, vi[4] - 1] = dnLabels["flat"]
            # T-wave
            vNewRhythm[0, vi[4] : vi[5]] = t * vhull(
                vi[5] - vi[4], fSkew=ts, fYa=ste / max(abs(t), 0.01)
            )
            if sAnomaly == "complete_Tinv":
                vNewRhythm[1, vi[4] : vi[5]] = dnLabels["T_inv"]
                vNewRhythm[3, vi[4]] = dnLabels["T_inv"]
                vNewRhythm[4, vi[5] - 1] = dnLabels["T_inv"]
            else:
                vNewRhythm[1, vi[4] : vi[5]] = dnLabels["T"]
                vNewRhythm[3, vi[4]] = dnLabels["T"]
                vNewRhythm[4, vi[5] - 1] = dnLabels["T"]
            # Final flat segment
            vNewRhythm[1, vi[5] :] = dnLabels["flat"]
            vNewRhythm[3, vi[5]] = dnLabels["flat"]
            vNewRhythm[4, -1] = dnLabels["flat"]

            # bradycardia
            if sAnomaly == "complete_brad":
                vNewRhythm[1, :] = dnLabels["bradycardia"]
                vNewRhythm[3, 0] = dnLabels["bradycardia"]
                vNewRhythm[4, -1] = dnLabels["bradycardia"]
            # tachycardia
            elif sAnomaly == "complete_tach":
                vNewRhythm[1, :] = dnLabels["tachycardia"]
                vNewRhythm[3, 0] = dnLabels["tachycardia"]
                vNewRhythm[4, -1] = dnLabels["tachycardia"]
            lSegments.append(vNewRhythm)

    # @profile
    def append_segs(sSegment, n, dParameters, vRVs, lSegments):
        # Complete ecg rhythms
        if "complete_" in sSegment:
            nSegs = 7
            # Determine mean duration
            if sSegment == "complete_brad":
                tMeanDur = dMeanDurs["complete_brad"]
            elif sSegment == "complete_tach":
                tMeanDur = dMeanDurs["complete_tach"]
            else:
                tMeanDur = dMeanDurs["complete_normal"]
            # Timings and durations
            vDurations, mTimings = rhythm_timings(
                n, vRVs[: nSegs * n], tMeanDur, dParameters
            )
            # Draw the parameters for the shape of the rhythms
            lParams = [
                dParameters[s]
                * (
                    1
                    + dParameters["stdev"] * vRVs[(i + nSegs) * n : (i + nSegs + 1) * n]
                )
                for i, s in enumerate(["P", "R", "SQ", "T", "P_skew", "T_skew"])
            ]
            if sSegment == "complete_noP":
                lParams[0] *= 0
            elif sSegment == "complete_Pinv":
                lParams[0] *= -1
            elif sSegment == "complete_Tinv":
                lParams[3] *= -1
            elif sSegment == "complete_noQRS":
                lParams[1] *= 0
            elif sSegment == "complete_Tinv":
                lParams[3] *= -1

            if sSegment == "complete_STelev":
                vSTel = dParameters["ST_elev"] * (
                    1 + dParameters["stdev"] * vRVs[13 * n : 14 * n]
                )
            elif sSegment == "complete_STdepr":
                vSTel = dParameters["ST_depr"] * (
                    1 + dParameters["stdev"] * vRVs[13 * n : 14 * n]
                )
            else:
                vSTel = np.zeros(n)
            append_rhythms(
                lSegments, zip(vDurations, mTimings, *lParams, vSTel), sSegment
            )
        # Single ecg segments
        else:
            # Treat P_inv and T_inv as inverted P and T waves, P_mitrale as notched P wave
            bInvert = False
            bNotch = False
            if sSegment in ["P_inv", "T_inv", "P_mitrale"]:
                if sSegment in ["P_inv", "T_inv"]:
                    bInvert = True
                elif sSegment == "P_mitrale":
                    bNotch = True
                sSegment = sSegment[0]
            # (Length of) flat segment has higher standard deviation
            fStdev = dParameters["stdev"]
            if sSegment == "flat":
                fStdev *= 2.5

            vtDurations = np.array(dMeanDurs[sSegment] * (1 + fStdev * vRVs[:n]), int)

            if sSegment in ["P", "T"]:
                vHeights = dParameters[sSegment] * (
                    1 + dParameters["stdev"] * vRVs[n : 2 * n]
                )
                vHeights *= (-1) ** int(bInvert)
                if sSegment in ["P", "T"]:
                    vSkews = dParameters[sSegment + "_skew"] * (
                        1 + dParameters["stdev"] * vRVs[2 * n :]
                    )
                    for d, h, s in zip(vtDurations, vHeights, vSkews):
                        vNewSeg = np.zeros((5, d))
                        vNewSeg[3, :] = np.nan
                        vNewSeg[4, :] = np.nan
                        vNewSeg[0] = h * vhull(
                            d, fSkew=s, fNotch=int(bNotch) * dParameters["P_mitrale"]
                        )
                        vNewSeg[1] = dnLabels[
                            sSegment + bInvert * "_inv" + bNotch * "_mitrale"
                        ]
                        vNewSeg[2] = dnLabels1["other"]
                        vNewSeg[3, 0] = dnLabels[
                            sSegment + bInvert * "_inv" + bNotch * "_mitrale"
                        ]
                        vNewSeg[4, -1] = dnLabels[
                            sSegment + bInvert * "_inv" + bNotch * "_mitrale"
                        ]
                        lSegments.append(vNewSeg)
            elif sSegment == "QRS":
                vHeights = dParameters["R"] * (
                    1 + dParameters["stdev"] * vRVs[n : 2 * n]
                )
                vSQ = dParameters["SQ"] * (1 + dParameters["stdev"] * vRVs[2 * n :])
                for d, h, s in zip(vtDurations, vHeights, vSQ):
                    vNewSeg = np.zeros((5, d))
                    vNewSeg[3, :] = np.nan
                    vNewSeg[4, :] = np.nan
                    vNewSeg[0] = h * vQRS(d, s)
                    vNewSeg[1] = dnLabels["QRS"]
                    vNewSeg[2] = dnLabels1["other"]
                    vNewSeg[3, 0] = dnLabels["QRS"]
                    vNewSeg[4, -1] = dnLabels["QRS"]
                    lSegments.append(vNewSeg)
            elif sSegment == "flat":
                for d in vtDurations:
                    vNewSeg = np.zeros((5, d))
                    vNewSeg[3, :] = np.nan
                    vNewSeg[4, :] = np.nan
                    vNewSeg[0] = np.zeros(d)
                    vNewSeg[1] = dnLabels["flat"]
                    vNewSeg[2] = dnLabels1["other"]
                    vNewSeg[3, 0] = dnLabels["flat"]
                    vNewSeg[4, -1] = dnLabels["flat"]
                    lSegments.append(vNewSeg)

    # Replace parameters with kwargs if present
    for p, v in dParameters.items():
        dParameters[p] = kwargs.get(p, v)
    if lRepertoire == "all":
        lRepertoire = (
            "P",
            "QRS",
            "T",
            "flat",
            "P_inv",
            "T_inv",
            "P_mitrale",
            "complete_normal",
            "complete_Tinv",
            "complete_STelev",
            "complete_Pinv",
            "complete_noP",
            "complete_noQRS",
            "complete_STdepr",
            "complete_tach",
            "complete_brad",
        )
    # Assign probabilites
    if dProbs is None:
        # Uniform distribution over segment/rhythm types
        lProbs = np.ones(len(lRepertoire)) / len(lRepertoire)
    else:
        if len(dProbs) < len(lRepertoire):
            # Probability of elements not in dProbs
            fSumP = np.sum(list(dProbs.values()))
            fDefaultP = np.clip((1. - fSumP) / (len(lRepertoire) - len(dProbs)), 0, 1)
        else:
            fDefaultP = 0
        lProbs = np.array([dProbs.get(s, fDefaultP) for s in lRepertoire])

    # Number of instances for each type of segment
    vnInsts = np.array(n * lProbs / np.sum(lProbs), int)
    # Increase random subset of these numbers so that their sum equals desired
    # total number of segments
    vnDev = n - np.sum(vnInsts)
    vnChange = np.random.choice(len(lRepertoire), size=np.abs(vnDev))
    vnInsts[vnChange] += np.sign(vnDev)

    # Determine total number of random variables needed and draw pool of RVs
    # Creating them all at once is much faster
    vnCumSumnRVs = np.cumsum(
        [0] + [vnInsts[i] * dnRVs[s] for i, s in enumerate(lRepertoire)]
    )
    spDistro = truncnorm(-3, 3)
    vRVs = spDistro.rvs(vnCumSumnRVs[-1])

    # Iterate over different segment types, append segments to list and shuffle
    lSegments = []
    for i, (s, inst) in enumerate(zip(lRepertoire, vnInsts)):
        # print(s)
        append_segs(
            s, inst, dParameters, vRVs[vnCumSumnRVs[i] : vnCumSumnRVs[i + 1]], lSegments
        )
    np.random.shuffle(lSegments)

    # Merge segments into one large array
    mSignal = np.hstack(lSegments)

    if fStdNoise > 0:
        mSignal[0,:] += np.random.randn(len(mSignal[0,:])) * fStdNoise

    return mSignal


def signal_and_target(
    nTrials: int,
    dProbs: dict,
    fHeartRate: float,
    tDt: float,
    strTargetMethod: str = "segment-extd",
    nMinWidth: int = 0,
    nMaxWidth: int = None,
    nWidth: int = 200,
    fScale: float = 1,
    bDetailled: bool = True,
    bVerbose: bool = False,
    fStdNoise: float = 0,
) -> (np.ndarray, np.ndarray):
    """
    signal_and_target - Produce training or test signal and target for network.
                        There are different methods for determining the target:
                            'rhythm' : Label complete rhythm where anomaly occurs
                            'segment-extd' : Label anomal segment and nWidth time
                                             points after it. Anomal part can be
                                             scaled ûsing fScale. Total label size
                                             can be clipped using nMinWidth and
                                             nMaxWidth.
    :param nTrials:         int Number of ECG rhythms
    :param dProbs:          dict Probabilities for different symptoms
    :param fHeartRate:      float Heart rate in rhythms per second
    :param tDt:             float time step size
    :param strTargetMethod: str Method for determining the target
                                Must be 'rhythm' or 'segment-extd'
    :param nMinWidth:       int Minimum anomal label length (only used
                                if strTargetMethod == 'segment-extd')
    :param nMaxWidth:       int Minimum anomal label length (only used
                                if strTargetMethod == 'segment-extd')
    :param nWidth:          int Number of timesteps to be marked as abnormal
                                after onset of anomaly (only used if
                                strTargetMethod == 'segment-extd')
    :param fScale:          float Scale width of labeled anomal segments
                                (only used if strTargetMethod == 'segment-extd')

    :param bDetailled:      bool Target contains info about anomaly type
    :param bVerbose:        bool Print information about generated signal
    :param fStdNoise:       float Standard deviation of Gaussian noise added to
                                  ECG signal
    :return:            1D-np.ndarray with ECG signal
                        2D-np.ndarray with target for each anomaly
    """

    # - Input
    vfECG, vnSegments, vnRhythms, vnSegStarts, vnSegEnds = labeled_signal(
        n=nTrials, fHeartRate=fHeartRate * tDt, dProbs=dProbs, fStdNoise=fStdNoise,
    )

    if strTargetMethod == "rhythm":
        # - Label complete rhythm where anomaly occurs
        vTarget = np.clip(vnRhythms, 0, 1)
        mTarget = vTarget.reshape(-1, 1)

        if bDetailled:
            mTarget = np.zeros((len(vfECG), 9), bool)
            for j in range(9):
                mTarget[:, j] = vnRhythms == j + 1

    elif strTargetMethod == "segment-extd":
        if bDetailled:

            mTarget = np.zeros((len(vfECG), 9), bool)

            # First 4 labels correspond to normal segments, others are anomalies (currently ignor p_mitrale)
            # Iterrate over anomal segements within rhythms
            for j in range(8):
                # First point of each anomaly
                viStarts = np.where(vnSegStarts == (j + 4))[0]
                # Final point of each anomaly
                viEnds = np.where(vnSegEnds == (j + 4))[0] + 1
                # Anomaly durations
                vnDurationsAnomalies = viEnds - viStarts
                # Scale durations, add fixed width, clip
                vnDurations = np.clip(
                    (vnDurationsAnomalies * fScale + nWidth), nMinWidth, nMaxWidth
                ).astype(int)
                # Include anomalies in target
                for iStart, nDur in zip(viStarts, vnDurations):
                    mTarget[iStart : iStart + nDur, j] = True
            # Isolated segments count as own anomaly type
            # First point of each isolated segment
            viStarts = np.where((np.isnan(vnSegStarts) == False) & (vnRhythms == 9))[0]
            # Final point of each anomaly
            viEnds = np.where((np.isnan(vnSegEnds) == False) & (vnRhythms == 9))[0] + 1
            # Segment durations
            vnDurationsAnomalies = viEnds - viStarts
            # Scale durations, add fixed width, clip
            vnDurations = np.clip(
                (vnDurationsAnomalies * fScale + nWidth), nMinWidth, nMaxWidth
            ).astype(int)
            # Include segments in target
            for iStart, nDur in zip(viStarts, vnDurations):
                mTarget[iStart : iStart + nDur, 8] = True

        else:  # somewhat obsolete
            mTarget = np.zeros((len(vfECG), 1), bool)
            viStarts = np.where(
                (vnSegStarts > 3) ^ ((vnRhythms == 9) & (vnSegStarts > 0))
            )[0]
            viEnds = np.where((vnSegEnds > 3) ^ ((vnRhythms == 9) & (vnSegEnds > 0)))[0]
            vnDurations = np.clip(
                ((viEnds - viStarts) * fScale + nWidth), nMinWidth, nMaxWidth
            ).astype(int)
            for iStart, nDur in zip(viStarts, vnDurations):
                mTarget[iStart : iStart + nDur, 0] = True

    if bVerbose:
        tDuration = vfECG.size * tDt
        print("Generated input and target")
        print(
            "\tLength of signal: {:.3f}s ({} time steps)\n".format(
                tDuration, vfECG.size
            )
        )

    return vfECG, mTarget


if __name__ == "__main__":
    # Brief unit test
    from matplotlib import pyplot as plt

    s = labeled_signal(1e2, 5e-3)  # , lRepertoire=['complete_brad'])
    plt.plot(s[0, :])
    plt.plot(s[1, :])
    plt.show()
