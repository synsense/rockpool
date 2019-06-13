#!/bin/bash

function rename () {
	echo "renaming $1 to $2"
	local FNS=$(grep --exclude=tags --exclude=TAGS --exclude-dir=.git --exclude-dir=build --exclude-dir=Build --exclude-dir=classes --exclude-dir=target--exclude-dir=Libraries --exclude=*.log --exclude=*~ --exclude=*.sh --exclude=*.min.js -rInE -l "\b$1\b")
        for fn in $FNS; do
	    	if test -e "$fn"; then
			echo $fn
	    		vim $fn -c '%s/\<'$1'\>/'$2'/g | wqa'
	    	fi
	done
}


rename requestQ request_q
rename resultQ result_q
rename numCores num_cores
rename nNumCores num_cores
rename fDelta_T delta_t
rename mfDelayIn delay_in
rename mfDelayRec delay_rec
rename fA a
rename fB b

rename _fA _a
rename _fB _b
rename _fDelta_T _delta_t
rename mfInputStep input_steps
rename vtTimeBase time_base
rename mnSpikeRaster spike_raster
rename mfInput inp
rename tStart t_start
rename tStop t_stop
rename vtTauN tau_mem
rename _vtTauN _tau_mem
rename vtTauS tau_syn
rename _vtTauS _tau_syn
rename vfVThresh v_thresh
rename _vfVThresh _v_thresh
rename vfVReset v_reset
rename _vfVReset _v_reset
rename vfVRest v_rest
rename _vfVRest _v_rest
rename vfCapacity capacity
rename tRefractoryTime refractory
rename _tRefractoryTime _refractory
rename vtTauW tau_w
rename _vtTauW _tau_w
rename bRecord record
rename vtEventTimes event_times
rename vtTimeBase time_base
rename vnEventChannels event_channels
rename mfRecordStates record_states
rename vnEventChannelOutput event_channel_out
rename vtEventTimeOutput event_time_out
rename nestProcess nest_process
rename vtNewTauN new_tau_mem
rename vtNewTauS new_tau_syn
rename vfBias bias
rename _vfBias _bias
rename vfNewBias new_bias
rename vfNewVThresh new_v_thresh
rename vfNewVReset new_v_reset
rename vfNewVRest new_v_rest
rename newFA new_a
rename newFB new_b
rename newFDelta_T new_delta_t
rename newVtTauW new_tau_w
rename ClassName class_name
rename vNewState new_state

rename vtTimeTrace time_trace
rename oInput inp
rename tupShape shape
rename sVariableName var_name
rename nTotalSize total_size
rename bAllowNone allow_none
rename mfNewW new_w
rename _mfW _weights
rename fNewNoiseStd new_noise_std
rename vbUseEvent use_event
rename getParam get_param
rename setParam set_param
rename tauN tau_mem
rename tRef refractory

rename vtTauSInp tau_syn_inp
rename vtTauSRec tau_syn_rec
rename eqNeurons neuron_eq
rename eqSynapses synapse_eq
rename strIntegrator integrator_name
rename _sggInput _input_generator
rename _ngLayer _neuron_group
rename _sgReceiver _inp_synapses
rename _sgRecurrentSynapses _rec_synapses
rename _spmReservoir _spike_monitor
rename _stmVmemIsyn _v_monitor
rename mfNoiseStep noise_step
rename taI_noise inp_noise
rename vfV v_state
rename vfIsynRec v_syn_rec
rename vfIsynInp v_syn_inp
rename fRangeV v_range
rename bKeepParams keep_params
rename vtTauSynR tau_syn_r

rename vfVBias bias
rename vfVSubtract v_subtract
rename vnIdMonitor monitor_id
rename aStateTimeSeries state_time_series
rename tCurrentTime t_now
rename vnIdOut id_out
rename bDebug debug
rename nIdOutIter id_out_iter
rename mfSpikeRaster spike_raster
rename vfNewThresh new_v_thresh
rename vfNewReset new_v_reset

rename vfVNew new_v_state
rename _vfVSubtract _v_subtract
rename _vfVBias _bias
rename tNewDt new_dt
rename _vnIdMonitor _id_monitor
rename vnNewIDs new_ids
rename mfInptSpikeRaster inp_spike_raster
rename liSpikeIDs spike_ids
rename ltSpikeTimes spike_times
rename bCNNWeights is_CNNWeights
rename dqvnNumRecSpikes num_rec_spikes_q
rename _dqvnNumRecSpikes _num_rec_spikes_q
rename lnTSSpikes ts_spikes

rename vnTSUntilRefrEnds ts_until_refr_ends
rename _vnTSUntilRefrEnds _ts_until_refr_ends
rename vnNumTSperRefractory ts_per_refr
rename _vnNumTSperRefractory _ts_per_refr
rename vbBias is_bias
rename dtypeState state_type
rename nStateMin min_state
rename _nStateMin _min_state
rename nStateMax max_state
rename _nStateMax _max_state
rename mfRecord record
rename rangeIterator range_iter
rename iCurrentTimeStep cur_time_step
rename vbInptSpikeRaster is_inp_spike_raster
rename vfUpdate update
rename vbRefractory is_refractory

rename vbSpiking is_spiking
rename vnNumRecSpikes num_rec_spikes

rename vtSpikeTimes spike_times
rename tseOut event_out
rename vtRecordTimes record_times
rename tscRecorded ts_recorded
rename nNumTSperDelay num_ts_per_delay
rename tNewBias new_bias
rename _nNumTSperBias _num_ts_per_bias
rename tNewDelay new_delay

rename lPrevSpikes prev_spiken
rename nDifference t_diff
rename vtRefractoryTime refractory
rename vtNewTime new_refractory
rename _dtypeState _state_type
rename dtypeNew new_type

rename tTauSyn tau_syn
rename _ngReceiver _neuron_group
rename _stmReceiver _state_monitor
rename vfIsyn syn_inp
rename vtTimeBaseOutput time_base_out
rename vbUseTime use_time
rename mfA a
rename tNewTau new_tau_syn


rename bAddEvents add_events
rename mnInputRaster inp_raster
rename mfWeightedInput weighted_input
rename mfNoise noise
rename _vStateNoBias _state_no_bias
rename vfKernel kernel
rename mfFiltered filtered
rename vConv conv
rename vEvents events
rename vConvShort conv_short
rename tsTarget ts_target
rename fRegularize regularize
rename fLearningRate learning_rate

rename mfWeighted weighted
rename mfOut out
rename mfXTX xtx
rename mfXTY xty
rename mfNewWeights new_weights
rename bFirst is_first
rename bFinal is_last
rename bStoreState store_states
rename bTrainBiases train_biases
rename mfTarget target

rename _mfXTY _xty
rename _mfXTX _xtx
rename nInputSize input_size
rename mfKahanCompXTY kahan_comp_xty
rename mfKahanCompXTX kahan_comp_xtx
rename mfUpdXTY upd_xty
rename mfUpdXTX upd_xtx
rename mfNewXTY new_xty
rename mfNewXTX new_xtx
rename _vTrainingState _training_state
rename mfSolution solution
rename nBatchSize batch_size

rename nEpochs epochs
rename nNumBatches num_batches
rename viSampleOrder sample_order
rename mfGradients gradients
rename viSampleIndices simple_indices

rename iEpoch ind_epoch
rename iBatch ind_batch
rename mfLinear linear
rename mfOutput output
rename nNumSamples num_samples
rename mfError error
rename _tTauSyn _tau_syn

rename inShape inp_shape
rename nKernels kernels
rename tFinal t_final
rename vbSpikeRaster spike_raster
rename mbOutRaster out_raster

rename vfWIn weights_in
rename dParamNeuron neuron_params
rename dParamSynapse syn_params
rename Imem i_mem
rename Iahp i_ahp
rename Ie_Recur i_ex_recur
rename Ii_Recur i_inh_recur

rename Ie_Recei i_ex_inp
rename Ii_Recei i_inh_inp
rename vfNewW new_weights

rename nNumInputEvents num_inp_events
rename mbInputChannelRaster inp_channel_raster
rename mnOutputChannelRaster out_channel_raster
rename vnRepetitions repetitions
rename vnChannelMask channel_mask
rename vnChannelsOut out_channels
rename vnNumOutputEventsPerInputEvent num_out_events_per_input_event
rename vtTimeTraceOut time_trace_out

rename tEnd t_end
rename vSpk input_times
rename vIdInput input_ids
rename aSpk spikes
rename nSpikeIndx spike_id
rename nInputId input_id
rename vbSpike has_spiked
rename vnSpike num_spikes
rename mfSpk spikes
rename evOut out_events
rename mfStateTimeSeries ts_state
rename _mfStateTimeSeries _ts_state



rename vfCleak leak
rename tSpikeDelay delay
rename tTauLeak tau_leak
rename _heapRemainingSpikes heap_remaining_spikes
rename tFirstLeak t_first_leak
rename nMaxNumLeaks max_num_leaks
rename vtLeak leak
rename _nLeakChannel _leak_channel
rename heapSpikes heap_spikes
rename vtRefractoryEnds t_refractory_ends
rename tTime t_time
rename mfWTotal weights_total
rename nLeakChannel leak_channel
rename vtRefr refractory
rename tDelay delay
rename lvStates states
rename ltTimes times
rename lnChannels channels

rename nChannel channel
rename vbNotRefractory is_not_refractory
rename vbStateBelowRest state_below_rest
rename vnSign sign
rename vbStillAboveThresh is_still_above_thresh
rename vtStopRefr t_stop_refr
rename viSpikeIDs l_spike_ids
rename nID n_id
rename tsRecorded ts_recorded
rename mStates states
rename _mfWTotal _weights_total
rename vfNewRest new_v_rest
rename vfNewLeak new_leak
rename _vtRefractoryTime _refractory

rename _tMinRefractory _min_refractory
rename tNewTauLeak new_tau_leak
rename _tTauLeak _tau_leak
rename _tSpikeDelay _delay
rename tNewSpikeDelay new_delay

rename fNoiseStd noise_std
rename _nTimeStep time_step
rename bVerbose verbose
rename mfNeuronInputStep neuron_inp_step
rename taI_inp inp_current
rename _nTimeStep _time_step
rename nNumSteps num_steps
rename nStep step
rename vbUseEvents use_events
rename _spmLayer _layer
rename tupInput inp
rename _stmVmem state_monitor
rename fLambda lambda_
rename vnTargetCounts target_counts
rename fEligibilityRatio eligibility_ratio
rename fMomentum momentum
rename nSize size
rename iSource source_id
rename vtEventTimesSource event_time_source


rename tSpkIn t_spike_in
rename vfVmem v_mem
rename mfEligibiity eligibility
rename miEligible is_eligible
rename nEligible eligible
rename vbUseEventOut use_out_events
rename viSpkNeuronOut spikes_out_neurons
rename vnSpikeCount spike_counts
rename iNeuron n_id
rename vfUpdates updates
rename _mfDW_previous _dw_previous
rename mfDW_current dw_current
rename iTarget target_id
rename nSizeIn size_in

rename isMultiple is_multiple
rename fTolerance tolerance
rename fMinRemainder min_remainder
rename iCurr curr
rename nTotal total
rename tPassed passed
rename fhReLu re_lu
rename vfX x
rename vCopy cop
rename mX x
rename fStdDev std_dev
rename mfCopy cop
rename fhActivation activation_func
rename vfGain gain
rename vfAlpha alpha
rename mfActivities activities
rename vDState d_state


rename vtTau tau
rename mfActivity activity
rename vfLambda lambda_
rename vfThisAct this_act
rename tMinTau min_tau
rename mSamplesAct sample_act
rename nEulerStepsPerDt euler_steps_per_dt
rename _fhActivation _activation
rename _mfKahanCompXTY _kahan_comp_xty
rename _vtTau _tau
rename vNewTau new_tau
rename _vfAlpha _alpha
rename vNewAlpha new_alpha
rename vNewBias new_bias
rename _vfGain _gain
rename vNewGain new_gain

rename mfInProcessed in_processed
rename nNumTimeStepsComb num_time_steps_comb
rename vtTimeComb time_comb
rename _nDelaySteps _delay_steps
rename mfSamplesComb samples_comb
rename nStepsIn steps_in
rename tsBuffer ts_buffer
rename mfSamplesOut samples_out
rename nDelaySteps delay_steps
rename mfBuffer buffer
rename vtNewTau new_tau
rename fhNewActivation new_activation


rename fVth thresh
rename __nIdMonitor__ __monitor_id__
rename _evOut _event_out
rename mfStateHistoryLog state_history_log
rename mfDataTimeStep data_time_step
rename mfStateHistoryLog state_history_log
rename mfSoftMax soft_max
rename tsOut ts_out


rename mfW_f weights_fast
rename mfW_s weights_slow
rename vtTauSynR_f tau_syn_r_fast
rename vtTauSynR_s tau_syn_r_slow
rename fhSpikeCallback spike_callback
rename _tMinTau _min_tau
rename tMinDelta min_delta
rename vtInputTimeTrace input_time_trace
rename mfStaticInput static_input
rename tFinalTime final_time
rename nSpikePointer spike_pointer
rename vtTimes times
rename mfDotV dot_v
rename nMaxSpikePointer max_spike_pointer
rename vnSpikeIndices spike_indices
rename vtRefractory vec_refractory
rename tLast t_last
rename VLast v_last
rename vfZeros zeros

rename vbSpikeIDs spike_ids
rename nNumSpikes num_spikes
rename vtSpikeDeltas spike_deltas
rename tSpikeDelta spike_delta
rename nFirstSpikeId first_spike_id
rename tShortestStep shortest_step
rename tSpike spike

rename nIntTime int_time
rename dotI_s_S dot_I_s_S
rename dotI_s_F dot_I_s_F
rename Syn_dotI syn_dot_I
rename Neuron_dotV neuron_dot_v
rename dotV dot_v
rename nExtend extend
rename mfV v
rename mfS s
rename mfF f
rename dResp resp

rename bUseHV use_hv
rename dSpikes spikes
rename _dLastEvolve _last_evolve
rename vfTauSynR_f tau_syn_r_f
rename __vfTauSynR_f __tau_syn_r_f
rename __vfTauSynR_s __tau_syn_r_s
rename vfTauSynR_s tau_syn_r_s
rename __vfVThresh __thresh
rename __vfVRest __rest
rename __vfVReset __reset
rename vnShape shape


rename vfData data
rename nMinLoc min_loc
rename fMinVal min_val
rename vbData data
rename fMin f_min
rename fMax f_max
rename fVal val
rename oData data

rename RepToNetSize rep_to_net_size
rename argwhere argwhere

rename fhReLU re_lu
rename vnSpikeIDs spike_ids


rename nNetSize net_size
rename mfGamma gamma

rename tDt dt
rename _nTimeStep _timestep
rename mfW weights
rename mfWIn weights_in
rename mfWRec weights_rec
rename strName name
rename vState state
rename nsize size
rename nsizeIn size_in
rename fNoiseStd noise_std
rename tsInput ts_input
rename tDuration duration
rename nNumTimesteps num_timesteps
rename bVerbose verbose
rename fTolRel tol_rel
rename fTolAbs tol_abs
rename cInput input_type
rename cOutput output_type



rename tDt dt
rename _nTimeStep _timestep
rename lEvolOrder evol_order
rename setLayers layerset
rename Method arguments:
rename lyrInput inputlayer
rename lyrOutput outputlayer
rename bExternalInput external_input
rename bVerbose verbose
rename lyrDel del_layer
rename lyrSource pre_layer
rename lyrTarget post_layer
rename fhTraining  training_fct
rename tsInput ts_input
rename tDuration duration
rename vtDurBatch batch_durs
rename nNumTimeSteps num_timesteps
rename vnNumTSBatch nums_ts_batch
rename bVerbose verbose
rename bHighVerbosity high_verbosity
rename fhStepCallback step_callback


rename fMu mu
rename fNu nu
rename tTauN tau_mem
rename tTauSynFast tau_syn_fast
rename tTauSynSlow tau_syn_slow
rename Omega_f omega_f
rename Omega_s omega_s
rename vfT_dash t_dash
rename Gamma_dash gamma_dash
rename vfT v_t
rename Omega_f_dash omega_f_dash
rename Omega_s_dash omega_s_dash

rename mfW_input weights_in
rename mfW_output weights_out
rename tTauSynO tau_syn_out
rename inputlayer input_layer
rename lyrReservoir reservoir_layer
rename outputlayer output_layer
rename netDeneve net_deneve


rename mfWInput weights_in
rename mfWRes weights_res
rename mfWOutput weights_out
rename vtTauInput tau_in
rename vtTauRes tau_res
rename vfBiasInput bias_in
rename vfBiasRes bias_res
rename fNoiseStdInput noise_std_in
rename fNoiseStdRes noise_std_res
rename fNoiseStdOut noise_std_out

rename nReservoirSize reservoir_size
rename nOutputSize output_size
rename BuildRandomReservoir build_random_reservoir
rename fInputWeightStd weights_in_std
rename fResWeightStd weights_res_std
rename fOutputWeightStd weights_out_std
rename fInputWeightMean weights_in_mean

rename fResWeightMean weights_res_mean
rename fOutputWeightMean weights_out_mean
rename BuildRateReservoir build_rate_reservoir

rename nDefaultMaxNumTimeSteps MAX_NUM_TIMESTEPS_DEFAULT
rename nEvolutionTimeStep evolution_timestep
rename mfNeuralInput neural_input
rename vSynapseState synapse_state
rename _vSynapseState _synapse_state
rename mfRecordSynapses synapse_recording
rename nRefractorySteps num_refractory_steps
rename _nRefractorySteps _num_refractory_steps
rename vnRefractoryCountdownSteps nums_refr_ctdwn_steps
rename _vnRefractoryCountdownSteps _nums_refr_ctdwn_steps
rename vtRefractoryCountdown t_refr_countdown
rename nMaxNumTimeSteps max_num_timesteps
rename _nMaxNumTimeSteps _max_num_timesteps
rename mfInputKernels matr_input_kernels
rename mbSpiking matr_is_spiking
rename vSynapseStateInp synapse_state_inp
rename _vSynapseStateInp _synapse_state_inp
rename nTimeStepStart timestep_start
rename iCurrentIndex idx_curr
rename mfCurrentInput matr_input_curr
rename nCurrNumTS num_ts_curr
rename vtRecTimesStates rec_times_states
rename tscRecStates ts_rec_states
rename tscRecSynapses ts_rec_synapses
rename vtRecTimesSynapses rec_times_synapses
rename vnSpikeTimeIndices spiketime_indices
rename vnChannels channels
rename mfKernels matr_kernels
rename nNumTSKernel num_ts_kernel
rename vtSpikeTimings spike_times
rename nStart n_start
rename nEnd n_end
rename essentialDict essential_dict
rename vfNewState new_state
rename nNewMax new_max

rename nTSRecurrent ts_recurrent
rename vfLeakRate leak_rate
rename _vfLeakRate _leak_rate
rename vfStateMin state_min
rename _vfStateMin _state_min
rename vLeakUpdate v_leak_update
rename vLeak v_leak
rename vfNewRate new_rate
rename vfNewMin new_min

rename bIntermediateResults calc_intermediate_results
rename _mfKahanCompXTX _kahan_comp_xtx
rename ctTarget ct_target
rename ctInput ct_input
rename ctWeights ct_weights
rename ctBiases ct_biases
rename ctSampleOrder ct_sample_order
rename ctSampleIndices ct_sample_indices
rename ctGradients ct_gradients
rename ctLinear ct_linear
rename ctOutput ct_output
rename ctError ct_error
rename _convSynapses conv_synapses
rename _convSynapsesTraining conv_synapses_training
rename nKernelSize kernel_size
rename mfInputKernelsTraining matr_input_kernels_training


rename tsrNumSpikes num_spikes
rename nInChannels num_in_channels
rename nOutChannels num_out_channels
rename outShape out_shape
rename fVThresh v_thresh
rename fVReset v_reset
rename fVSubtract v_subtract
rename tsrState state
rename tsrConvOut conv_out
rename tsrIn tsr_in
rename tsrInReshaped tsr_in_reshaped
rename tsrInput tsr_input
rename nBatch n_batch
rename lyrTorch lyr_torch
rename _lyrTorch _lyr_torch
rename lyrNewTorch lyr_torch_new
