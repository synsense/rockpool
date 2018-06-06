import numpy as np 
from scipy import sparse
from matplotlib import pyplot as plt
plt.ion()

import TimeSeries as ts 
import ecg
import network as nw 
from layers.recurrent import rate as rec
from layers.feedforward import rate as ff


tDt = 0.005
nTrialsTr = 1000
nTrialsTe = 100
fHeartRate = 1		# Heart rate in rhythms per second

nDimIn = 1
nDimOut = 1

nResSize = 256
tTau = 50*tDt

fRegularize = 0.001

kwResWeights = {'nResSize' : nResSize,
				'fConnectivity' : 0.4,
				'bPartitioned' : True,
				'fRatioExc' : 0.5,
				'fNormalize' : 0.95}

dProbs = {'complete_normal' : 0.8,
          'complete_noP' : 0.025,
          'complete_Pinv' : 0.025,
          'complete_noQRS' : 0.025,
          'complete_Tinv' : 0.025,
          'complete_STelev' : 0.025,
          'complete_STdepr' : 0.025,
          'complete_tach' : 0.025,
          'complete_brad' : 0.025}

def rec_weights(nResSize,
                fConnectivity,
                bPartitioned=False,
                fRatioExc=0.5,
                global_scale = 1,
                inh_scale=1,
                fNormalize=0.95):
    """ Set recurrent weights for the reservoir """
    n_connect = int(fConnectivity*nResSize**2)
    # Choosing coordinates is a bit tedious as choice only supports 1d arrays
    rows = np.repeat(np.arange(nResSize), nResSize)
    cols = np.tile(np.arange(nResSize), nResSize)
    choices = np.random.choice(nResSize*nResSize, size=n_connect, replace=False)
    if bPartitioned:
        values = np.abs(np.random.randn(n_connect))
        n_exc = int(nResSize*fRatioExc)
        # Values for first n_exc columns are positive, rest negative
        inhibitory = cols[choices] >= n_exc
        values *= (-inh_scale)**inhibitory * global_scale / np.sqrt(nResSize)
    else:
        values = 2*np.random.rand(n_connect) - 1
    # Weight matrix
    weights = sparse.csr_matrix((
        values, (rows[choices], cols[choices])), shape=(nResSize,nResSize))
    eigenvalues = np.linalg.eigvals(weights.toarray())
    if fNormalize is not None:
        # Normalize weights so that spectral radius is fNormalize
        weights *= fNormalize/np.amax(np.abs(eigenvalues))
    return weights

def signal(nTrials: int,
		   dProbs: dict,
		   fHeartRate: float,
		   tDt: float,
		   nTargetWidth: int = 200,
		   bVerbose: bool = False) -> (np.ndarray, np.ndarray):
    """
    signal - Produce training or test signal and target for network.
    		 Target starts with anomaly and has a fixed length.
    :param nTrials: 	int Number of ECG rhythms
    :dProbs: 			dict Probabilities for different symptoms
    fHeartRate: 		float Heart rate in rhythms per second
    :tDt: 				float time step size
    :nTargetWidth: 		int Number of timesteps to be marked as abnormal
    						after onset of anomaly
    :bVerbose: 			bool Print information about generated signal
    :return: 			2 1D-np.ndarrays ECG signal and target
    """

    # - Input
    vfECG, vnSegments, vnRhythms, vAnomStarts = ecg.labeled_input(n=nTrials,
                               fHeartRate=fHeartRate*tDt,
                               dProbs=dProbs)

    # - Label interval of fixed length after onset of anomaly

    #   Any ECG segment that is not abnormal by itself (label in mInput[1] either 1, 2 or 3)
    #   and that is part of a complethe ECG rhythm (label in mInput[2] is not 9) is considered
    #   normal. Isolated segments (mInputs[2]==9) still count as anomalies.
    vAnomStarts[((vAnomStarts==1) ^ (vAnomStarts==2) ^ (vAnomStarts==3)) & (vnRhythms!=9)] = 0
    v0AnomStarts = np.r_[np.zeros(nTargetWidth), vAnomStarts]      # vAnomStarts with leading 0s
    vTargetFix = np.array([(v0AnomStarts[i:i+nTargetWidth+1] != 0).any()
                           for i in range(len(vAnomStarts))])

    if bVerbose:
        tDuration = vfECG.size*tDt
        print('Generated input and target')
        print('\tLength of signal: {:.3f}s ({} time steps)\n'.format(tDuration, vfECG.size))

    return vfECG, vTargetFix

# - Weights
mfW_in = 2*np.random.rand(nDimIn, nResSize)
mfW_res = rec_weights(**kwResWeights).toarray()

# - Layers
flIn = ff.PassThrough(mfW=mfW_in, tDt=tDt, tDelay=0, strName='in')
rlRes = rec.RecRateEuler(mfW=mfW_res, vtTau=tTau, tDt=tDt, strName='res')
flOut = ff.PassThrough(mfW=np.zeros((nResSize, nDimOut)), tDt=tDt, tDelay=0, strName='out')

# - Network
net = nw.Network(flIn, rlRes, flOut)

# - Training signal
vfEcgTr, vfTgtTr = signal(nTrials=nTrialsTr, dProbs=dProbs, fHeartRate=fHeartRate, tDt=tDt, bVerbose=True)
vtTimeTr = np.arange(0, vfEcgTr.size*tDt, tDt)
tsInTr = ts.TimeSeries(vtTimeTr, vfEcgTr)
tsTgtTr = ts.TimeSeries(vtTimeTr, vfTgtTr)

# - Training
def cTrain(net, dtsSignal, bFirst, bFinal):
	tsInput = dtsSignal[flOut.lyrIn.strName]
	tsTarget = tsTgtTr.resample(tsInput.vtTimeTrace)
	flOut.train_rr(tsTarget, tsInput, fRegularize, bFirst, bFinal)

net.train(cTrain, tsInTr, tDurBatch=100)
net.reset_all()

# - Sanity check with training signal
dTr = net.evolve(tsInTr)
net.reset_all()
tsResTr = dTr[rlRes.strName]
plt.plot(tsResTr.mfSamples@flOut.mfW+flOut.vfBias)
plt.plot(tsTgtTr.mfSamples)
plt.plot(0.2*tsInTr.mfSamples, color='k', alpha=0.5, zorder=-1)
	

# - Test signal
vfEcgTe, vfTgtTe = signal(nTrials=nTrialsTe, dProbs=dProbs, fHeartRate=fHeartRate, tDt=tDt, bVerbose=True)
vtTimeTe = np.arange(0, vfEcgTe.size*tDt, tDt)
tsInTe = ts.TimeSeries(vtTimeTe, vfEcgTe)
tsTgtTe = ts.TimeSeries(vtTimeTe, vfTgtTe)

# - Test
dTe = net.evolve(tsInTe)
net.reset_all()
tsResTe = dTe[rlRes.strName]
plt.figure()
plt.plot(tsResTe.mfSamples@flOut.mfW+flOut.vfBias)
plt.plot(tsTgtTe.mfSamples)
plt.plot(0.2*tsInTe.mfSamples, color='k', alpha=0.5, zorder=-1)