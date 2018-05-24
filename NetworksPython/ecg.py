from scipy.stats import truncnorm
import numpy as np


# - Functions for constructing the ecg segments

def swave(t,t0,t1,y0,y1):
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
    return 0.5*(y0+y1+(y0-y1)*np.cos(np.pi*(t-t0)/(t1-t0)))*(t>=t0)*(t<=t1)

def vhull(tLength,fYa=0,fYb=0,fSkew=0, fNotch=0):
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
    fP = 1+abs(fSkew)
    if fSkew >= 0:
        t0 = 0
        t1 = tLength
    else:
        t0 = tLength
        t1 = 0
    vt = np.arange(tLength)
    # Bump, where first and last points are 0, extremum is 1, shifted
    # away from center if fP != 1
    vBump = 0.5*(1.-np.cos(2.*np.pi*np.abs((vt-t0)/(t1-t0))**fP))
    # Location of extremum
    tExtr = int(0.5**(1./fP)*(t1-t0)+t0)
    # Add notch
    nHalfNotchWidth = int(0.25*tLength)
    vNotch = -0.5 * fNotch * (1.-np.cos(np.pi/nHalfNotchWidth*vt[:2*nHalfNotchWidth]))
    vBump[tExtr-nHalfNotchWidth:tExtr+nHalfNotchWidth] += vNotch
    # Change the values of first and last points, such that derivative there
    # is still 0 and curve remains smooth
    vElevL = swave(vt,0,tExtr,fYa,0)
    vElevR = swave(vt,tExtr,tLength,0,fYb)
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
    t0 = int(tLength/4)
    t1 = int(3*tLength/4)
    # First minimum
    sig[:t0] = - np.sin(4./tLength*np.pi*vt[:t0])*(vt[:t0]/tLength)**4
    # Second minimum
    sig[t1:] = - np.sin(4./tLength*np.pi*(tLength-vt[t1:]))*((tLength-vt[t1:])/tLength)**4
    # Large spike
    sig[t0:t1] = 1./128. * np.sin(2.*np.pi/tLength*(vt[t0:t1]-tLength/4.))
    # Normalize so that peak is 1.
    sig /= np.amax(np.abs(sig))
    # Change the values of first and last points, such that derivative there
    # is still 0 and curve remains smooth
    sig += (swave(vt, 0, 0.5*tLength, fYa, 0)
           + swave(vt, 0.5*tLength, tLength, 0, fYb))
    return sig


def hull(t,ta,tb,ya=0,yb=0,skew=0):
    p = 1+abs(skew)
    if skew >= 0:
        t0 = ta
        t1 = tb
    else:
        t0 = tb
        t1 = ta
    #np.seterr(invalid='ignore')
    bump = 0.5*(t>=ta)*(t<=tb)*(1.-np.cos(2.*np.pi*np.abs((t-t0)/(t1-t0))**p))
    #np.seterr(invalid='warn')
    tp = 0.5**(1./p)*(t1-t0)+t0
    elev_l = swave(t,ta,tp,ya,0)
    elev_r = swave(t,tp,tb,0,yb)
    return (bump+elev_l+elev_r)*(t>=ta)*(t<=tb) + ya*(t<ta) + yb*(t>tb)


def fQRS(t,ta, tb, SQ, ya=0, yb=0):
    delta = tb-ta
    x = t-ta
    sig =( (x<delta/4.)*(-1)*np.sin(4./delta*np.pi*x)*(x/delta)**4
          +(x>3.*delta/4.)*(-1)*np.sin(4./delta*np.pi*(delta-x))*((delta-x)/delta)**4
          +(x>=delta/4.)*(x<=3.*delta/4.)*1./128.*np.sin(2.*np.pi/delta*(x-delta/4.)))
    return ( 50.31*sig*(1.+SQ*x/delta)*(t<tb)*(t>ta)
             + swave(t, ta, 0.5*(ta+tb), ya, 0)
             + swave(t, 0.5*(ta+tb), tb, 0, yb))


#Heartbeats
def hb(t, P=0.5, R=5, SQ=3, T=1,
       t1=1, t2=2.75, t3=3, t4=4, t5=5, t6=7,
       P_skew = 0.5, T_skew = 0.5, PQ_elev=0, ST_elev=0 ):
    """Produce a single period of a generic ECG signal

    The signal is represented as a continuous, differntiable function
    in time.
    t:              time, either as scalar or as ndarray.
    P, T, R:        maxima of the P- and T-waves and of the QRS complex.
    SQ:             difference between the minima S and Q, with S=(1+SQ)*Q.
    P_skew, T_skew: asymmetry of P and T waves. Lean towards right (left)
                    for positive (negative) values.
    PQ_elev, ST_elev: elevation of PQ- or ST-elements
    t1 .. t6 set the times of:
        t1: beginning of P-R interval
        t2: beginning of P-R segment
        t3: beginning of QRS interval
        t4: beginning of S-T segment
        t5: end of S-T segment (beginning of T-wave)
        t6: end of Q-T interval
    The length of the segment after the Q-T interval corresponds to the
    number of timepoints after t6, i.e. for any t>t6 the function will
    return 0.
    So far the U-wave is omitted but could be easily added if considered
    relevant.
    Terms as in http://www.bem.fi/book/15/15.htm, Fig. 15.4
    """

    sP = P*hull(t,t1,t2, yb=PQ_elev/max(P,0.01), skew=P_skew)
    sQRS = R*fQRS(t, t3, t4, SQ, PQ_elev/max(R,0.01), ST_elev/max(R,0.01))
    sT = T*hull(t,t5,t6, ya=ST_elev/max(T,0.01), skew=T_skew)
    sig = ( (t>t1)*(t<t2)*sP
            + (t>t2)*(t<t3)*PQ_elev
            + (t>t3)*(t<t4)*sQRS
            + (t>t4)*(t<t5)*ST_elev
            + (t>t5)*(t<t6)*sT
          )
    return sig


# @profile
def labeled_input(n, fHeartRate, lRepertoire='all', dProbs=None, **kwargs):
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
        kwargs : dict to modify default parameters that are otherwise 
                 defined in dParameters
    """
    n = int(n)
    dParameters = {'t1' : 0.165,
                   't2' : 0.25,
                   't3' : 0.375,
                   't4' : 0.49,
                   't5' : 0.575,
                   't6' : 0.775,
                   'P' : 0.5,
                   'P_skew' : 0.5,
                   'P_mitrale' : 0.5,
                   'R' : 5,
                   'SQ' : 3,
                   'T' : 1,
                   'T_skew' : 0.5,
                   'ST_elev' : 0.9,
                   'ST_depr' : -0.8,
                   'stdev' : 0.1}
    dnLabels = {'flat' : 0,
                'P' : 1,
                'QRS' : 2,
                'T' : 3,
                'P_inv' : 4,
                'P_mitrale' : 5,
                'no_P' : 6,
                'no_QRS' : 7,
                'T_inv' : 8,
                'ST_elev' : 9,
                'ST_depr' : 10,
                'bradychardia' : 11,
                'tachychardia' : 12}
    # Second-stage labels
    dnLabels1 = {'complete_normal' : 0,
                 'complete_noP' : 1,
                 'complete_Pinv' : 2,
                 'complete_noQRS' : 3,
                 'complete_Tinv' : 4,
                 'complete_STelev' : 5,
                 'complete_STdepr' : 6,
                 'complete_tach' : 7,
                 'complete_brad' : 8,
                 'other' : 9}
    # Mean durations for each type of rhythm or segment
    dMeanDurs = {'P' : (dParameters['t2']-dParameters['t1'])/fHeartRate,
                 'QRS' : (dParameters['t4']-dParameters['t3'])/fHeartRate,
                 'T' : (dParameters['t6']-dParameters['t5'])/fHeartRate,
                 'flat' : (1-dParameters['t6'])/fHeartRate,
                 'complete_normal' : 1./fHeartRate,
                 'complete_brad' : 2./fHeartRate,
                 'complete_tach' : 1./fHeartRate/2}
    # For each type of rhythm or segment, the number of random variables needed
    # This way RVs can be drawn at once, which is more efficient
    dnRVs = {'P' : 3, # duration, skew, height
             'P_inv' : 3, # duration, skew, height
             'P_mitrale' : 3, # duration, skew, height
             'QRS' : 3, # duration, QS-ratio, height
             'T' : 3, # duration, skew, height
             'T_inv' : 3, # duration, skew, height
             'flat' : 1, # duration
             'complete_normal' : 13, # 7 durations, rest for P,QRS,T
             'complete_noP' : 13, # 7 durations, rest for P,QRS,T
             'complete_Pinv' : 13, # 7 durations, rest for P,QRS,T
             'complete_noQRS' : 13, # 7 durations, rest for P,QRS,T
             'complete_Tinv' : 13, # 7 durations, rest for P,QRS,T
             'complete_STelev' : 14, # 7 durations, rest for P,QRS,T
             'complete_STdepr' : 14, # 7 durations, rest for P,QRS,T
             'complete_tach' : 13, # 7 durations, rest for P,QRS,T
             'complete_brad' : 13} # 7 durations, rest for P,QRS,T
    
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
        vDurations = np.array(tMeanDur * (1 + dParams['stdev']*vRVs[:n]), int)
        # Mean timings within single rhythm of length 1
        vMeanTimings = np.r_[0, [dParams['t{}'.format(t)] for t in range(1,7)]]
        # Drawing intervals instead of points in time avoids overlapping of segments
        vMeanIntervals = np.diff(vMeanTimings)
        # nx7 matrix of intervals from 0 to t1, t1 to t2,..,t6 to end
        mIntervals = np.array([m * (dParams['stdev']*vRVs[i*n:(i+1)*n] + 1)
                               for i, m in enumerate(vMeanIntervals)]).T
        mTimings = np.cumsum(mIntervals, axis=1)
        mTimings *= vDurations.reshape(-1,1)
        return vDurations, np.asarray(mTimings, dtype=int)

    # @profile
    def append_rhythms(lSegments, zParameters, sAnomaly=None):
        """Create ecg rhythms and append them to lSegments. Shapes are
        determined by zParameters, labels are according to sAnomaly.
            lSegments : list to which new rhythms are appended
            zParameters : zip of arrays of parameters needed to determine
                          the shape of each rhythm: durations, timings,
                          P, R, SQ, T, Pskew, Tskew, STelevation
        """
        for d, vi, p, r, q, t, ps, ts, ste in zParameters:
            vNewRhythm = np.zeros((3,d))
            # Indicate if normal rhythm in third row of array
            vNewRhythm[2,:] = dnLabels1[sAnomaly]
            # Initial flat segment
            vNewRhythm[1,:vi[0]] = dnLabels['flat']
            # P-wave
            vNewRhythm[0,vi[0]:vi[1]] = p*vhull(vi[1]-vi[0], fSkew=ps)    
            if sAnomaly == 'complete_Pinv':
                vNewRhythm[1,vi[0]:vi[1]] = dnLabels['P_inv']
            elif sAnomaly == 'complete_noP':
                vNewRhythm[1,vi[0]:vi[1]] = dnLabels['no_P']
            else :
                vNewRhythm[1,vi[0]:vi[1]] = dnLabels['P']
            # PQ segment
            vNewRhythm[1,vi[1]:vi[2]] = dnLabels['flat']               
            # QRS complex
            vNewRhythm[0,vi[2]:vi[3]] = r*vQRS(vi[3]-vi[2], q, fYb=ste/max(abs(r), 0.01))
            if sAnomaly == 'complete_noQRS':
                vNewRhythm[1,vi[2]:vi[3]] = dnLabels['no_QRS']
            else:
                vNewRhythm[1,vi[2]:vi[3]] = dnLabels['QRS']
            # ST segment
            if sAnomaly == 'complete_STelev':
                vNewRhythm[0,vi[3]:vi[4]] = ste
                vNewRhythm[1,vi[3]:vi[4]] = dnLabels['ST_elev']       
            elif sAnomaly == 'complete_STdepr':
                vNewRhythm[0,vi[3]:vi[4]] = ste
                vNewRhythm[1,vi[3]:vi[4]] = dnLabels['ST_depr']
            else:
                vNewRhythm[1,vi[3]:vi[4]] = dnLabels['flat']
            # T-wave
            vNewRhythm[0,vi[4]:vi[5]] = t*vhull(vi[5]-vi[4], fSkew=ts, fYa=ste/max(abs(t), 0.01))   
            if sAnomaly == 'complete_Tinv':
                vNewRhythm[1,vi[4]:vi[5]] = dnLabels['T_inv']
            else:
                vNewRhythm[1,vi[4]:vi[5]] = dnLabels['T']
            # Final flat segment
            vNewRhythm[1,vi[5]:] = dnLabels['flat']
            if sAnomaly == 'complete_brad':
                vNewRhythm[1,:] = dnLabels['bradychardia']
            elif sAnomaly == 'complete_tach':
                vNewRhythm[1,:] = dnLabels['tachychardia']
            lSegments.append(vNewRhythm)

    # @profile
    def append_segs(sSegment, n, dParameters, vRVs, lSegments):
        # Complete ecg rhythms
        if 'complete_' in sSegment:
            nSegs = 7
            # Determine mean duration
            if sSegment == 'complete_brad':
                tMeanDur = dMeanDurs['complete_brad']
            elif sSegment == 'complete_tach':
                tMeanDur = dMeanDurs['complete_tach']
            else: 
                tMeanDur = dMeanDurs['complete_normal']
            # Timings and durations
            vDurations, mTimings = rhythm_timings(n, vRVs[:nSegs*n], tMeanDur, dParameters)
            # Draw the parameters for the shape of the rhythms
            lParams = [dParameters[s] * (1 + dParameters['stdev']*vRVs[(i+nSegs)*n:(i+nSegs+1)*n])
                       for i, s in enumerate(['P', 'R', 'SQ', 'T', 'P_skew', 'T_skew'])]
            if sSegment == 'complete_noP':
                lParams[0] *= 0
            elif sSegment == 'complete_Pinv':
                lParams[0] *= -1
            elif sSegment == 'complete_Tinv':
                lParams[3] *= -1
            elif sSegment == 'complete_noQRS':
                lParams[1] *= 0
            elif sSegment == 'complete_Tinv':
                lParams[3] *= -1
            
            if sSegment == 'complete_STelev':
                vSTel = dParameters['ST_elev'] * (1 + dParameters['stdev']*vRVs[13*n:14*n])
            elif sSegment == 'complete_STdepr':
                vSTel = dParameters['ST_depr'] * (1 + dParameters['stdev']*vRVs[13*n:14*n])
            else:
                vSTel = np.zeros(n)
            append_rhythms(lSegments,
                           zip(vDurations, mTimings, *lParams, vSTel),
                           sSegment)
        # Single ecg segments
        else:
            # Treat P_inv and T_inv as inverted P and T waves, P_mitrale as notched P wave
            bInvert = False
            bNotch = False
            if sSegment in ['P_inv', 'T_inv', 'P_mitrale']:
                if sSegment in ['P_inv', 'T_inv']:
                    bInvert = True
                elif sSegment == 'P_mitrale':
                    bNotch = True
                sSegment = sSegment[0]
            # (Length of) flat segment has higher standard deviation
            fStdev = dParameters['stdev']
            if sSegment == 'flat':
                fStdev *= 2.5

            vtDurations = np.array(dMeanDurs[sSegment] * (1 + fStdev*vRVs[:n]), int)

            if sSegment in ['P', 'T']:
                vHeights = dParameters[sSegment] * (1 + dParameters['stdev']*vRVs[n:2*n])
                vHeights *= (-1)**int(bInvert)
                if sSegment in ['P', 'T']:
                    vSkews = dParameters[sSegment+'_skew'] * (1 + dParameters['stdev']*vRVs[2*n:])
                    for d, h, s in zip(vtDurations, vHeights, vSkews):
                        vNewSeg = np.zeros((3, d))
                        vNewSeg[0] = h*vhull(d, fSkew=s, fNotch=int(bNotch)*dParameters['P_mitrale'])
                        vNewSeg[1] = dnLabels[sSegment + bInvert*'_inv' + bNotch*'_mitrale']
                        vNewSeg[2] = dnLabels1['other']
                        lSegments.append(vNewSeg)
            elif sSegment == 'QRS':
                vHeights = dParameters['R'] * (1 + dParameters['stdev']*vRVs[n:2*n])
                vSQ = dParameters['SQ'] * (1 + dParameters['stdev']*vRVs[2*n:])
                for d, h, s in zip(vtDurations, vHeights, vSQ):
                    vNewSeg = np.zeros((3, d))
                    vNewSeg[0] = h*vQRS(d, s)
                    vNewSeg[1] = dnLabels['QRS']
                    vNewSeg[2] = dnLabels1['other']
                    lSegments.append(vNewSeg)
            elif sSegment == 'flat':
                for d in vtDurations:
                    vNewSeg = np.zeros((3, d))
                    vNewSeg[0] = np.zeros(d)
                    vNewSeg[1] = dnLabels['flat']
                    vNewSeg[2] = dnLabels1['other']
                    lSegments.append(vNewSeg)
        
    #Replace parameters with kwargs if present
    for p, v in dParameters.items():
        dParameters[p] = kwargs.get(p, v)
    if lRepertoire == 'all':
        lRepertoire = ('P', 'QRS', 'T', 'flat', 'P_inv', 'T_inv', 'P_mitrale',
                       'complete_normal', 'complete_Tinv', 'complete_STelev',
                       'complete_Pinv',  'complete_noP', 'complete_noQRS',
                       'complete_STdepr', 'complete_tach', 'complete_brad')
    if dProbs is None:
        # Uniform distribution over segment/rhythm types
        lProbs = np.ones(len(lRepertoire))/len(lRepertoire)
    else:
        if len(dProbs) < len(lRepertoire):
            # Probability of elements not in dProbs
            fSumP = np.sum(list(dProbs.values()))
            fDefaultP = np.clip((1.-fSumP)/(len(lRepertoire)-len(dProbs)), 0, 1)
        else:
            fDefaultP = 0
        lProbs = np.array([dProbs.get(s, fDefaultP) for s in lRepertoire])
    
    # Number of instances for each type of segment
    vnInsts = np.array(n*lProbs/np.sum(lProbs), int)
    # Increase random subset of these numbers so that their sum equals desired
    # total number of segments
    vnDev = n - np.sum(vnInsts)
    vnChange = np.random.choice(len(lRepertoire), size=np.abs(vnDev))
    vnInsts[vnChange] += np.sign(vnDev)

    # Determine total number of random variables needed and draw pool of RVs
    # Creating them all at once is much faster
    vnCumSumnRVs = np.cumsum([0] + [vnInsts[i]*dnRVs[s] for i,s in enumerate(lRepertoire)])
    spDistro = truncnorm(-3, 3)
    vRVs = spDistro.rvs(vnCumSumnRVs[-1])

    # Iterate over different segment types, append segments to list and shuffle
    lSegments = []
    for i, (s, inst) in enumerate(zip(lRepertoire, vnInsts)):
        # print(s)
        append_segs(s, inst, dParameters, vRVs[vnCumSumnRVs[i]:vnCumSumnRVs[i+1]], lSegments)
    np.random.shuffle(lSegments)

    # Merge segments into one large array
    mSignal = np.hstack(lSegments)

    return mSignal
    

def hbr(n_points, t1=0.165, t2=0.25, t3=0.375, t4=0.49, t5=0.575, t6=0.775, **kwargs):
    """Wrapper for hb where number of timepoints of an ecg period is
    provided as argument n_points.

    Timings t1..t6 can be provided relative(!) to this number, i.e.
    t6=1 would mean that the end of the Q-T coincides with the end
    of the signal.
    P, R, T, SQ are be passed on to hb as kwargs.
    """

    x = np.linspace(0,1,n_points)
    return hb(x, t1=t1,t2=t2,t3=t3,t4=t4,t5=t5,t6=t6,**kwargs)



def anomal_beat(length=200, symptom='any', stdev=0.1):

    #Normal distro, truncated at +- 3*sigma
    distro = truncnorm(-3, 3)
    dur = int(length * (1+stdev*distro.rvs(1)))
    #Means of intervals between timings 0,t1..t6 (not normalized)
    meantimes = np.array([130,70,100,90,70,160,180])
    #Means for P, R, T and SQ
    meanparas = np.array([0.1, 1, 0.2, 3])

    signal = np.zeros(dur)
    intervals = meantimes*(1.+distro.rvs(len(meantimes))*stdev)
    #Normalized timings
    times = np.cumsum(intervals)/np.sum(intervals)
    paras = meanparas*(1.+distro.rvs(len(meanparas))*stdev)
    #Format kwargs for timings
    kwargs = {'t{}'.format(j+1) : t for j,t in enumerate(times[:-1])}
    #Add kwargs for peak parameters
    d_paras = {p : param for p,param in zip(['P','R','T','SQ'],paras)}
    kwargs.update(d_paras)

    #Anomalies
    repertoire = ('T_inv', 'P_inv', 'no_P', 'ST_elev', 'ST_depr', 'P_block', 'long_PR')
    if symptom == 'any':
        symptom = np.random.choice(repertoire)
    if symptom == 'T_inv':
        kwargs['T'] = -kwargs['T']
    elif symptom == 'P_inv':
        kwargs['P'] = -kwargs['P']
    elif symptom == 'ST_elev':
        kwargs['ST_elev'] = 0.19*(1+stdev*distro.rvs(1))
    elif symptom == 'ST_depr':
        kwargs['ST_elev'] = -0.15*(1+stdev*distro.rvs(1))
    elif symptom == 'P_block':
        kwargs['R'] = 0
    elif symptom == 'no_P':
        kwargs['P'] = 0
    elif symptom == 'long_PR':
        delta1 = 0.75*kwargs['t1']
        kwargs['t1'] -= delta1
        kwargs['t2'] -= delta1
        delta2 = 0.5*(1-kwargs['t6'])
        for i in range(3,7):
            kwargs['t'+str(i)] += delta2
    #Create ecg period and add it to signal
    return hbr(n_points=dur,**kwargs)


# @profile
def dist_signal(n, length=None, stdev=0.1, p_normal=1, repertoire='all'):

    #Normal distro, truncated at +- 3*sigma
    distro = truncnorm(-3, 3)
    #Distribution of numbers of timepoints for each signal period

    if length is not None:
        m_mean = length//n
        sigma = 0.2*m_mean
        m = np.array(sigma * distro.rvs(n) + m_mean, int)
        #Set duration of last pulse so that signal has given total length
        remaining = length - np.sum(m[:-1])
        m[-1] = np.clip(remaining, 1, None)
    else:
        m_mean = 1200
        sigma = 200
        m = np.array(sigma * distro.rvs(n) + m_mean, int)

    #Means of intervals between timings 0,t1..t6 (not normalized)
    meantimes = np.array(([130,70,100,90,70,160,180]))
    #Means for P, R, T and SQ
    meanparas = np.array([0.1, 1, 0.2, 3])

    healthy = np.zeros(np.sum(m))
    abnormal = np.zeros(np.sum(m))
    labels = np.zeros(np.sum(m))
    start = 0

    #Anomalies
    if repertoire == 'all':
        repertoire = ('T_inv', 'P_inv', 'no_P', 'ST_elev', 'ST_depr', 'P_block', 'long_PR', 'normal')
    #If 'normal' is not in repertoire, add it to the end
    elif 'normal' not in repertoire:
        repertoire = tuple(repertoire) + ('normal',)
    #If it already is, make sure it is at the end
    else:
        repertoire = list(repertoire)
        repertoire.append(repertoire.pop(repertoire.index('normal')))
    p_ano = (1-p_normal)/(len(repertoire)-1)
    p = p_ano*np.ones(len(repertoire))
    p[-1] = p_normal
    symptom_list = []

    #Iterate over signal periods and draw random variables
    for i,d in enumerate(m):
        intervals = meantimes*(1.+distro.rvs(len(meantimes))*stdev)
        #Normalized timings
        times = np.cumsum(intervals)/np.sum(intervals)
        paras = meanparas*(1.+distro.rvs(len(meanparas))*stdev)
        #Format kwargs for timings
        kwargs = {'t{}'.format(j+1) : t for j,t in enumerate(times[:-1])}
        #Add kwargs for peak parameters
        d_paras = {p : param for p,param in zip(['P','R','T','SQ'],paras)}
        kwargs.update(d_paras)

        #Create one period of healthy signal
        healthy[start:start+d] = hbr(n_points=d,**kwargs)

        #Create one period of abnormal signal with modified parameters in kwargs
        symptom = np.random.choice(repertoire, p=p)
        symptom_list.append(symptom)
        if symptom == 'T_inv':
            kwargs['T'] = -kwargs['T']
        elif symptom == 'P_inv':
            kwargs['P'] = -kwargs['P']
        elif symptom == 'ST_elev':
            kwargs['ST_elev'] = 0.19*(1+stdev*distro.rvs(1))
        elif symptom == 'ST_depr':
            kwargs['ST_elev'] = -0.15*(1+stdev*distro.rvs(1))
        elif symptom == 'P_block':
            kwargs['R'] = 0
        elif symptom == 'no_P':
            kwargs['P'] = 0
        elif symptom == 'long_PR':
            delta1 = 0.75*kwargs['t1']
            kwargs['t1'] -= delta1
            kwargs['t2'] -= delta1
            delta2 = 0.5*(1-kwargs['t6'])
            for i in range(3,7):
                kwargs['t'+str(i)] += delta2

        abnormal[start:start+d] = hbr(n_points=d,**kwargs)
        if symptom != 'normal':
            labels[start:start+d] = 1

        start += d

    starttimes = np.hstack((0, np.cumsum(m[:-1])))

    return healthy, abnormal, symptom_list, starttimes, labels



def rhrt(n, length=None, mean_dur=1200, cut=False, sigma=0.1):
    """Plot n periods of the ecg defined in hb with variable timings
    and peak heights, which are drawn from a truncated Gaussian, the
    parameters of which need to be set in the function itself.
    In a next step the means (m_mean, meantimes, meanparas) and
    standard deviations (sigma, stdev) could be provided as arguments
    to rhrt.
    """

    #Normal distro, truncated at +- 3*sigma
    distro = truncnorm(-3, 3)
    #Distribution of numbers of timepoints for each signal period
    if length is not None:
        mean_dur = length//n
        m = np.array((sigma*distro.rvs(n) + 1) * mean_dur, int)
        #Set duration of last pulse so that signal has given total length
        remaining = length - np.sum(m[:-1])
        m[-1] = np.clip(remaining, 1, None)
    else:
        m = np.array((sigma*distro.rvs(n) + 1) * mean_dur, int)

    #Means of intervals between timings 0,t1..t6 (not normalized)
    meantimes = np.array(([130,70,100,90,70,160,180]))
    #Means for P, R, T and SQ
    meanparas = np.array([0.1, 1, 0.2, 3])
    #Same standard deviation for all intervals and peak heights

    signal = np.zeros(np.sum(m))
    start = 0
    #Iterate over signal periods and draw random variables
    for i,d in enumerate(m):
        intervals = meantimes*(1.+distro.rvs(len(meantimes))*sigma)
        #Normalized timings
        times = np.cumsum(intervals)/np.sum(intervals)
        paras = meanparas*(1.+distro.rvs(len(meanparas))*sigma)
        #Format kwargs for timings
        kwargs = {'t{}'.format(j+1) : t for j,t in enumerate(times[:-1])}
        #Add kwargs for peak parameters
        d_paras = {p : param for p,param in zip(['P','R','T','SQ'],paras)}
        kwargs.update(d_paras)
        #Create ecg period and add it to signal
        period = hbr(n_points=d,**kwargs)
        signal[start:start+d]=period
        start += d

    if cut:
        return signal[:length]
    else:
        return signal


# Might be useful at some point
def remove_consecutive_duplicates(vA):
    """All except one element of each sequence of consecutive
    duplicates in an array are removed. Apart from that the original
    structure is maintained. For instance [a,a,a,b,b,c,a,a,d] becomes
    [a,b,c,a,d].
    """

    # - Only keep elements that are non equal to the following one
    vB = np.r_[vA[1:], np.nan]
    return vA[np.where(vA != vB)]



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    s = segments(1e2, 5e-3)#, lRepertoire=['complete_brad'])
    plt.plot(s[0,:])
    plt.plot(s[1,:])
    plt.show()