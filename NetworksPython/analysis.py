import numpy as np
from matplotlib import pyplot as plt
from bokeh import plotting as bk


def analyze(vOutput, vTarget, vInput, fThrDetect, nWindow=0, nClose=200, bVerbose=True, bPlot=True):
    """Analyze output signal vOutput wrt to target: 
        - Sensitivity: Which of the anomalies have been detected?
            Detected means that there is at least one detection during
              or up to nWindow time steps after the anomaly
            A detection is when vOutput crosses fThrDetect from below
              and there has been no detection within the last nClose
              time steps  
            If two anomalies happen with less than
              nClose time steps between their beginning, detecting
              both is unlikely. Therefore in this case it is sufficient
              if vOutput is above threshold.
        - False detections: Each detection that happens within a normal
          interval counts as false detection.
            A normal interval is an interval where no anomaly has
              occured within the previous nWindow time steps.
        - For the specificity, each normal interval in which one or
          more detections happen counts as false positive
    """

    nDuration = len(vOutput)
    vbAboveThr = (np.abs(vOutput) > fThrDetect).flatten()

    vnDetectInds, = np.where(np.logical_and(vbAboveThr,                             # Above threshold at current point
                                            np.r_[False, vbAboveThr[:-1]]==False))  # Below threshold at previous point
    # vnDetectEnds = np.where(np.logical_and(vbAboveThr,                          # Above threshold at previous point
    #                                        1-np.r_[vbAboveThr[1:], 0]))[0] + 1  # Below threshold at curent point
    # If Threshold crossed from below within nClose time steps after previous
    # crossing from below, do not count as additional detection
    np.seterr(invalid='ignore')
    vnIgnoreDetects = vnDetectInds - np.r_[np.nan, vnDetectInds[:-1]] < nClose
    np.seterr(invalid='warn')
    vnDetectInds = vnDetectInds[vnIgnoreDetects == False]
    # vnDetectEnds = vnDetectEnds[np.r_[1-vnIgnoreDetects[1:], True]]   # Remove ends preciding the removed start
    vbDetects = np.repeat(False, nDuration)
    vbDetects[vnDetectInds] = True
    
    # - Sensitivity
    vnAnomStarts, = np.where(np.logical_and(vTarget == 1,                     # Current value is 1
                                            np.r_[0, vTarget[:-1]] == 0))     # Previous value is 0
    vnAnomEnds = np.where(np.logical_and(vTarget == 1,                        # Previous value is 1
                                         np.r_[vTarget[1:], 0] == 0))[0] + 1  # Current value is 0
    lbDetectedAnom = [vbDetects[min(s, nDuration-1):min(e, nDuration-1)].any()
                      for s, e in zip(vnAnomStarts, vnAnomEnds+nWindow)]
    # Anomalies that occur close together (with less than nWindow time steps between)
    # Only require vOutput to be above threshold, no new detect
    np.seterr(invalid='ignore')
    vnAnomCloseStartsInd, = np.where(vnAnomStarts - np.r_[np.nan, vnAnomStarts[:-1]] < nClose)
    np.seterr(invalid='warn')
    for i in vnAnomCloseStartsInd:
        if vbAboveThr[vnAnomStarts[i]:min(vnAnomEnds[i]+nClose, nDuration)].any():
            lbDetectedAnom[i] = True
    nTruePos = np.sum(lbDetectedAnom)
    nFalseNeg = len(lbDetectedAnom) - nTruePos
    fSensitivity = nTruePos / len(lbDetectedAnom)
    
    # - False positives and specificity
    # How many detections are in each normal interval
    # Normal interval: No anomaly within previous nWindow time points
    lnFalseDetects = [np.sum(vbDetects[min(s, nDuration-1):max(s, min(e, nDuration-1))])
                      for s, e in zip(np.r_[0, vnAnomEnds+nWindow],        # Iterate over intervals between anomalies
                                      np.r_[vnAnomStarts, nDuration])]
    # Number of normal intervals with no detection
    nTrueNeg = np.sum(np.asarray(lnFalseDetects) == 0)
    nFalseDetects = np.sum(lnFalseDetects)
    fSpecificity = nTrueNeg / len(lnFalseDetects)

    nErrors = nFalseNeg + nFalseDetects

    v0Target = np.r_[np.zeros(nWindow), vTarget]     # vTarget with leading 0s
    vbAnomal = np.array([(v0Target[i:i+nWindow+1] >= 0.5).any() for i in range(len(vTarget))])
        
    if bVerbose:
        print('Sensitivity : {:.1%} ({} anomalies)'.format(fSensitivity, len(lbDetectedAnom)))
        print('Specificity : {:.1%} ({} normal intervals)'.format(fSpecificity, len(lnFalseDetects)))
        print('{} Errors: \n {} of {} anomalies missed, {} false detections'.format(nErrors,
                                                                                    nFalseNeg,
                                                                                    len(lnFalseDetects),
                                                                                    nFalseDetects))
    if bPlot:
        # - Plotting
        fig, ax = plt.subplots()
        ax.plot(vOutput)
        ax.plot(vTarget)
        ax.plot([0, nDuration], [fThrDetect, fThrDetect], 'k--')
        ax.plot(0.2*vInput, lw=2, color='k', alpha=0.4, zorder=-1)
        # Higlight missed anomalies red, detected anomalies green and false positive intervals red
        for i, (s, e) in enumerate(zip(vnAnomStarts, vnAnomEnds)):
            if lbDetectedAnom[i]:
                # ax.fill_between(np.arange(s, e+nWindow), -1, 1, color='g', alpha=0.2)
                ax.plot([s, e+nWindow], [1.5, 1.5], color='g', lw=15, alpha=0.5)
            else:
                # ax.fill_between(np.arange(s, e+nWindow), -1, 1, color='r', alpha=0.2)
                ax.plot([s, e+nWindow], [1.5, 1.5], color='r', lw=15, alpha=0.5)
        for i, (s, e) in enumerate(zip(np.r_[0, vnAnomEnds+nWindow], 
                                       np.r_[vnAnomStarts, nDuration])):
            if lnFalseDetects[i] > 0.1:
                # ax.fill_between(np.arange(s, e), -1.5, -1, color='r', alpha=0.2)
                ax.plot([s, e], [-0.75,-0.75], color='r', lw=15, alpha=0.5)
        # Mark false detections with vertical red line
        # Indicate time points during and up to nWindow time steps after anomaly
        for i in np.where(np.logical_and(vbAnomal == False, vbDetects)):
            try:
                ax.plot([i,i], [-0.75, 1.5], 'r--', lw=3, alpha=0.5)
            except ZeroDivisionError:
                pass
    
        plt.show()

    return {'vnAnomStarts' : vnAnomStarts,
            'vnAnomEnds' : vnAnomEnds,
            'vnDetectInds' : vnDetectInds,
            'vbDetects' : vbDetects,
            'vbAnomal' : vbAnomal,
            'lbDetectedAnom' : lbDetectedAnom,
            'lnFalseDetects' : lnFalseDetects,
            'nTruePos' : nTruePos,
            'nFalseNeg' : nFalseNeg,
            'nTrueNeg' : nTrueNeg,
            'nFalseDetects' : nFalseDetects,
            'fSensitivity' : fSensitivity,
            'fSpecificity' : fSpecificity,
            'nErrors' : nErrors}

    
def find_threshold(vOutput, vTarget, nWindow, nClose, nAttempts, fMin=0, fMax=1, nRecursions=2):
    """Find a threshold that minimizes errors wrt vOutput and vTarget.
    Search on a grid with nAttempts steps. If nRecursions>0, continue
    search on grid around previous optimum.
        vOutput : Network output
        vTarget : Target output
        nWindow : Number of time steps after which an anomaly can still be detected
        nAttempts : Number of values to try for threshold within one run
        fMin, fMax : lowest, largest value of values to be tested
        nRecursion : How many times search should be refined on smaller-scale grid
    """

    vGrid = np.linspace(fMin, fMax, nAttempts)
    vErrors = np.array([analyze(vOutput, vTarget, None, t, nWindow, nClose, False, False)['nErrors']
                        for t in vGrid])
    nIndMin = np.argmin(vErrors)

    fThrOpt = vGrid[nIndMin]
    # print('Best threshold so far: {} ({} errors) \nRemaining recursions: {}'.format(
    #     fThrOpt, vErrors[nIndMin], nRecursions))
    if nRecursions > 0:
        return find_threshold(vOutput, vTarget, nWindow, nClose, nAttempts,
                              fMin=vGrid[max(0, nIndMin-1)],
                              fMax=vGrid[min(nIndMin+2, nAttempts-1)],
                              nRecursions=nRecursions-1)
    else:
        # print('Best threshold found: {} ({} errors)'.format(fThrOpt, vErrors[nIndMin]))
        return fThrOpt


def find_coincidental_normals(vInput0, mTarget0, mTarget1, tMeanDur):
    """Check if by coincidence there are unmarked, normal ecg rhythms
    in the input. This can be useful if the input signal is constructed
    from many single segments (instead of complete rhythms). Complete
    rhythms will be detected as 'possibly normal' unless they are 
    bradycarid or tachycardic or if anomalies are labelld in detail in
    second stage (bDetailledAnomalies1 == True).
        vInput0 : Input to first reservoir
        mTarget0 : Labels for first reservoir in binary matrix form
        mTarget1 : Labels for second reservoir in binary matrix form
        tMeanDur : Expected duration of a normal rhythm (in time steps)
    """

    vTarget0Int = np.where(mTarget0)[1] # mTarget0 in (non-binary) vector form
    vTarget1Int = np.where(mTarget1)[1] # mTarget0 in (non-binary) vector form
    vSegments, vSTInds, vSTReps, vSTRec = remove_consecutive_duplicates(vTarget0Int)
    vNormalSequence = np.array([0,1,0,2,0,3,0])
    nLenNS = len(vNormalSequence)
    mIndices = np.arange(len(vSegments)-nLenNS+1)[:,None] + np.arange(nLenNS)
    # vFound is True if at the corresponding position in vSegments an occurence of vNormalSequence begins
    vFound = (vSegments[mIndices] == vNormalSequence).all(1)
    vFound = np.r_[vFound, (nLenNS-1)*[False]]
    # Indices where vInput0 corresponds to vNormalSequence
    lvIndFound = [np.arange(vSTInds[min(i, len(vSTInds)-1)],
                            vSTInds[min(i+nLenNS, len(vSTInds)-1)]) 
                  for i in np.where(vFound)[0]]
    # Corresponding values
    lvMatches = [vInput0[i] for i in lvIndFound]
    # Iterate over these matches. If Their length corresponds to a normal ecg rhythm and they are
    # not tagged as such, plot them and ask if they should be tagged as normal
    lIndsUnflagged = []
    for i, (vMatch, vInd) in enumerate(zip(lvMatches, lvIndFound)):
        if ((0.75*tMeanDur <= len(vMatch) <= 1.4*tMeanDur
              and not vTarget1Int[vInd].any())):
            # plt.plot(vMatch)
            # plt.show()
            # b.append(input('If this looks like a correct ecg rhythm press 1 and then enter, otherwise just enter \n'))
            # plt.close()
            lIndsUnflagged.append(i)
    print('There are {}  unflagged but possibly normal rhythms ({:.1%})'.format(
        len(lIndsUnflagged), len(lIndsUnflagged)/len(lvMatches)))

    return np.array(lvMatches)[lIndsUnflagged], np.array(lvIndFound)[lIndsUnflagged]

    
def remove_consecutive_duplicates(vA):
    """All except one element of each sequence of consecutive
    duplicates in an array are removed. Apart from that the original
    structure is maintained. For instance [a,a,a,b,b,c,a,a,d] becomes
    [a,b,c,a,d].
    Return the resulting sequence vValues, the (first) indices vIndices
    of these elements in the array, the number of repetitions for each
    vRepetitions element and the indices of the original array elements
    within vValues, so that vValues = vA[vIndices]. vA can be
    reconstructed with vA = vValues[vReconstruct] or
    vA = np.array([v for v, r in zip(vValues, vRepetitions) for i in range(r)])
    """

    # - Only keep elements that are non equal to the following one
    vB = np.r_[vA[1:], np.nan]
    vC = np.r_[np.nan, vA[:-1]]
    vIndices = np.where(vA != vC)[0]
    vValues = vA[vIndices]
    vRepetitions = np.diff(np.r_[vIndices, len(vA)])
    vReconstruct = np.array([j for j, r in enumerate(vRepetitions) for i in range(r)])
    return vValues, vIndices, vRepetitions, vReconstruct


def plot(vInput, dOut, ddAn, dfThr, fDt, sTitle=''):

    dAnLoc = ddAn['local']
    dAnBrd = ddAn['broad']
    dAnFix = ddAn['fix']

    fig = bk.figure(title=sTitle,
                    plot_height=400,
                    x_axis_label='Time in s',
                    y_axis_label='Output',
                    x_range=[0,5],
                    y_range=[-1.5,2],
                    tools='pan, xwheel_zoom, xwheel_pan, box_zoom, reset, save')
    fig.line(fDt*np.arange(len(vInput)), 0.3*vInput, legend='Input', color='grey', alpha=0.5, line_width=3)
    fig.line(fDt*np.arange(len(dOut['local'])), dOut['local'].flatten(), legend='Only anomalies labelled', color='blue')
    fig.line(fDt*np.arange(len(dOut['broad'])), dOut['broad'].flatten(), legend='Complete rhythms labelled', color='orange')
    fig.line(fDt*np.arange(len(dOut['fix'])), dOut['fix'].flatten(), legend='Fix length after onset', color='maroon')
    # Indicate anomalies
    mAnomalyStartEndLoc = fDt*np.vstack((dAnLoc['vnAnomStarts'],
                                         dAnLoc['vnAnomEnds'])).T
    mAnomalyStartEndBrd = fDt*np.vstack((dAnBrd['vnAnomStarts'],
                                         dAnBrd['vnAnomEnds'])).T
    mAnomalyStartEndFix = fDt*np.vstack((dAnFix['vnAnomStarts'],
                                         dAnFix['vnAnomEnds'])).T
    # Hit by compl rhythm
    fig.multi_line(mAnomalyStartEndBrd[dAnBrd['lbDetectedAnom']].tolist(),
                   (-0.5*np.ones((np.sum(dAnBrd['lbDetectedAnom']), 2))).tolist(),
                   line_width=38,
                   alpha=0.5,
                   color='green')
    # Missed by compl rhythm
    fig.multi_line(mAnomalyStartEndBrd[np.array(dAnBrd['lbDetectedAnom'])==False].tolist(),
                   (-0.5*np.ones((np.sum(np.array(dAnBrd['lbDetectedAnom'])==False), 2))).tolist(),
                   line_width=38,
                   alpha=0.5,
                   color='red')
    # Hit by fix interval
    fig.multi_line(mAnomalyStartEndFix[dAnFix['lbDetectedAnom']].tolist(),
                   (-1*np.ones((np.sum(dAnFix['lbDetectedAnom']), 2))).tolist(),
                   line_width=38,
                   alpha=0.5,
                   line_dash=(12,4),
                   color='green')
    # Missed by fix interval
    fig.multi_line(mAnomalyStartEndFix[np.array(dAnFix['lbDetectedAnom'])==False].tolist(),
                   (-1*np.ones((np.sum(np.array(dAnFix['lbDetectedAnom'])==False), 2))).tolist(),
                   line_width=38,
                   alpha=0.5,
                   line_dash=(12,4),
                   color='red')
    # Hit by local label
    fig.multi_line(mAnomalyStartEndLoc[dAnLoc['lbDetectedAnom']].tolist(),
                   (1.5*np.ones((np.sum(dAnLoc['lbDetectedAnom']), 2))).tolist(),
                   line_width=38,
                   alpha=0.5,
                   # line_dash=(12,4),
                   color='green')
    # Missed by local label
    fig.multi_line(mAnomalyStartEndLoc[np.array(dAnLoc['lbDetectedAnom'])==False].tolist(),
                   (1.5*np.ones((np.sum(np.array(dAnLoc['lbDetectedAnom'])==False), 2))).tolist(),
                   line_width=38,
                   alpha=0.5,
                   # line_dash=(12,4),
                   color='red')
    # False detections local label
    vnFalseDetectsLoc = fDt*np.where(np.logical_and(dAnLoc['vbDetects'], 
                                                    dAnLoc['vbAnomal']==False))[0]
    fig.multi_line(np.vstack((vnFalseDetectsLoc, vnFalseDetectsLoc)).T.tolist(),
                   np.vstack((2*np.ones_like(vnFalseDetectsLoc),
                              0.5*np.ones_like(vnFalseDetectsLoc))).T.tolist(),
                   color='red',
                   line_dash='dashed',
                   line_width=3)
    # False detections fix interval
    vnFalseDetectsFix = fDt*np.where(np.logical_and(dAnFix['vbDetects'], 
                                                    dAnFix['vbAnomal']==False))[0]
    fig.multi_line(np.vstack((vnFalseDetectsFix, vnFalseDetectsFix)).T.tolist(),
                   np.vstack((-0.5*np.ones_like(vnFalseDetectsFix),
                              -1.5*np.ones_like(vnFalseDetectsFix))).T.tolist(),
                   color='red',
                   line_dash='dashed',
                   line_width=3)
    # False detections  compl rhythm
    vnFalseDetectsBrd = fDt*np.where(np.logical_and(dAnBrd['vbDetects'], 
                                                    dAnBrd['vbAnomal']==False))[0]
    fig.multi_line(np.vstack((vnFalseDetectsBrd, vnFalseDetectsBrd)).T.tolist(),
                   np.vstack((0.5*np.ones_like(vnFalseDetectsBrd),
                              -0.5*np.ones_like(vnFalseDetectsBrd))).T.tolist(),
                   color='red',
                   line_dash='dashed',
                   line_width=3)
    # Thresholds
    fig.line([0, fDt*len(vInput)], [dfThr['local'], dfThr['local']],
             line_dash='dashed', color='blue', legend='Thr Local', line_width=2)
    fig.line([0, fDt*len(vInput)], [dfThr['broad'], dfThr['broad']],
             line_dash='dashed', color='orange', legend='Thr Compl.', line_width=2)
    fig.line([0, fDt*len(vInput)], [dfThr['fix'], dfThr['fix']],
             line_dash='dashed', color='maroon', legend='Thr Fix Int.', line_width=2)
    
    return fig


class PlotResultsAnomalies():
    def __init__(self, dInputs, dResultsAn,  ddAnalysisAnom, dThr, fDt):
        dsAnomaliesSL = {'complete_Tinv' : 'Inverted T wave',
                         'complete_Pinv' : 'Inverted P wave',
                         'complete_noP' : 'Missing P wave',
                         'complete_STelev' : 'Elevated ST segment',
                         'complete_STdepr' : 'Depressed ST segment',
                         'complete_noQRS' : 'Missing QRS complex',
                         'complete_tach' : 'Tachycardia',
                         'complete_brad' : 'Bradycardia'}
        lsAnomalies = list(dInputs.keys())
        lsNets = list(dThr.keys())
        ddOutputs = {ano : {net : dResultsAn[(net, ano)]
                            for net in lsNets}
                     for ano in lsAnomalies}
        ddAnalysis = {ano : {net : ddAnalysisAnom[(net, ano)]
                             for net in lsNets}
                      for ano in lsAnomalies}
        self.ddAnalysisAnom = ddAnalysisAnom
        self.dBkPlots = {ano : plot(dInputs[ano], ddOutputs[ano], ddAnalysis[ano],
                                    dThr, fDt, sTitle=dsAnomaliesSL[ano])
                         for ano in lsAnomalies}

    def __call__(self, sAnom):
        """Print results for anomalies and return plot"""
        dAnLoc = self.ddAnalysisAnom[('local', sAnom)]
        dAnBrd = self.ddAnalysisAnom[('broad', sAnom)]
        dAnFix = self.ddAnalysisAnom[('fix', sAnom)]
        print('Only anomalies Labelled:')
        print('Sensitivity : {:.1%} (in total: {} anomalies)'.format(
              dAnLoc['fSensitivity'], len(dAnLoc['lbDetectedAnom'])))
        print('Specificity : {:.1%} (in total: {} normal intervals)'.format(
              dAnLoc['fSpecificity'], len(dAnLoc['lnFalseDetects'])))
        print('{} Errors: \n {} of {} anomalies missed, {} false detections'.format(
              dAnLoc['nErrors'], dAnLoc['nFalseNeg'], len(dAnLoc['lnFalseDetects']), dAnLoc['nFalseDetects']))
        
        print('\nComplete rhythms labelled:')
        print('Sensitivity : {:.1%} (in total: {} anomalies)'.format(
              dAnBrd['fSensitivity'], len(dAnBrd['lbDetectedAnom'])))
        print('Specificity : {:.1%} (in total: {} normal intervals)'.format(
              dAnBrd['fSpecificity'], len(dAnBrd['lnFalseDetects'])))
        print('{} Errors: \n {} of {} anomalies missed, {} false detections'.format(
              dAnBrd['nErrors'], dAnBrd['nFalseNeg'], len(dAnBrd['lnFalseDetects']), dAnBrd['nFalseDetects']))

        print('\nFixed length labels:')
        print('Sensitivity : {:.1%} (in total: {} anomalies)'.format(
              dAnFix['fSensitivity'], len(dAnFix['lbDetectedAnom'])))
        print('Specificity : {:.1%} (in total: {} normal intervals)'.format(
              dAnFix['fSpecificity'], len(dAnFix['lnFalseDetects'])))
        print('{} Errors: \n {} of {} anomalies missed, {} false detections'.format(
              dAnFix['nErrors'], dAnFix['nFalseNeg'], len(dAnFix['lnFalseDetects']), dAnFix['nFalseDetects']))

        return self.dBkPlots[sAnom]