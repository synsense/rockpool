# Change log

All notable changes between Rockpool releases will be documented in this file

## Unreleased

### Added
* Added a cycles model for Xylo A and Xylo IMU, enabling to calculate the required master clock frequency for Xylo
* Added support for NIR, for importing and exporting Rockpool torch networks

### Changed
* `LIFExodus` now supports vectors as threshold parameter

### Fixed
* `TypeError` when using `LIFExodus`
* Update `jax.config` usage
* Power measurement for `xyloA2` was not considering AFE channels
* Removed `check_grads` from Jax tests, since this will fail for LIF neurons due to surrograte gradients
* Fixed a bug in `AFESim` on windows, where the maximum int32 value would be exceeded when seeding the AFE simulation
* Fixed stochasticity in some unit tests
* Fixed a bug in `channel_quantize`, where quantization would be incorrectly applied for Xylo IMU networks with Nien < Nhid


### Deprecated

* Brian2 tests are not running -- Brian2 backend will be soon removed

### Removed

### Security


## [v.2.7.1 hotfix] -- 2024-01-19

### Fixed

* Bug in Xylo IMU mapper, where networks with more than 128 hidden neurons could not be mapped


## [v2.7] -- 2023-09-25

### Added


* Dependency on `pytest-random-order` v1.1.0 for test order randomization
* New HowTo tutorial for performing constrained optimisation with torch and jax
* Xylo IMU application software support:

  * `mapper`, `config_from_specification` and graph mapping support
  * `XyloSim` module: SNN core simulation for Xylo IMU
  * `IMUIFSim` module: Simulation of the input encoding interface with sub-modules:
    * `BandPassFilter`
    * `FilterBank`
    * `RotationRemoval`
    * `IAFSpikeEncoder`
    * `ScaleSpikeEncoder`
  * `XyloIMUMonitor` module: Real-time hardware monitoring for Xylo IMU
  * `XyloSamna` module: Interface to the SNN core
  * `IMUIFSamna` module: Interface to `IMUIF`, utilizing neurons in the SNN core
  * `IMUData` module: Collection of sensor data from the onboard IMU sensor
  * Utility functions for network mapping to the Xylo IMU HDK, interfacing, and data processing
  * Introductory documentation providing an overview of Xylo IMU and instructions on configuring preprocessing
* New losses, with structure similar to PyTorch
  * PeakLoss which can be imported as `peak_loss = rockpool.nn.losses.PeakLoss()`
  * MSELoss which can be imported as  `mse_loss = rockpool.nn.losses.MSELoss()`

### Changed

* Update dependency version of pytest-xdist to >=3.2.1.
* Update to `Sequential` API. `Sequential` now permits instantiation with an `OrderedDict` to specify module names. `Sequential` now supports an `.append()` method, to append new modules, optionally specifying a module name.
* Cleaned up tree manipulation libraries and added to documentation. Implemented unit tests.
* Removed obsolete unit tests
* Changed semantics of transformation configurations for QAT, to only include attributes which will be transformed, rather than all attributes. This fixes an incompatibility with torch >= 2.0.
* Added support for latest `torch` versions
* New fine-grained installation options
* Renamed power measurement dict keys returned by Xylo Audio 2 (`syns61201`) `XyloSamna` module, to be more descriptive
* Upgrade minimum Python version supported to 3.8
* Upgrade minimum JAX version supported to 0.4.10
* Rearranged Xylo documentation to separate overview, Xylo Audio and Xylo IMU

### Fixed

* Fixed bug in initialising access to MC3620 IMU sensor on Xylo IMU HDK, where it would fail with an error the on the second initialisation

### Deprecated

### Removed

* NEST backend completely removed
* Removed spiking output surrogate "U" from LIF modules

### Security

## [v2.6] -- 2023-03-22

### Added

* Dynap-SE2 Application Software Support (jax-backend)

  * jax backend `DynapSim` neuron model with its own custom surrogate gradient implementation
  * `DynapSamna` module handling low-level HDK interface under-the-hood
  * rockpool network <-> hardware configuration bi-directional conversion utilities
    * Network mapping: `mapper()` and `config_from_specification()`
    * sequentially combined `LinearJax`+`DynapSim` network getters : `dynapsim_net_from_config()` and `dynapsim_net_from_spec()`
  * transistor lookup tables to ease high-level parameters <-> currents <-> DAC (coarse, fine values) conversions
  * Dynap-SE2 specific auto-encoder quantization `autoencoder_quantization()`
    * Custom `DigitalAutoEncoder` implementation and training pipeline
  * `samna` alias classes compensating the missing documentation support
  * unit tests + tutorials + developer docs
  * `DynapseNeuron` graph module which supports conversion from and to `LIFNeuronWithSynsRealValue` graph
  * hardcoded frozen and dynamic mismatch prototypes
  * mismatch transformation (jax)
* `LIFExodus` now supports training time constants, and multiple time constants
* Improved API for `LIFTorch`
* Implemented `ExpSynExodus` for accelerated training of exponential synapse modules
* Added initial developer documentation
* Added MNIST tutorial
* Fixed notebook links to MyBinder.org


### Changed

* Updated Samna version requirement to >=0.19.0
* User explicitly defines Cuda device for LIFExodus, ExpDynExodus and LIFMembraneExodus
* Improved error message when a backend is missing
* Improved transient removal in `syns61201.AFESim`

### Fixed

* Weight scaling was too different for output layers and hidden layers.
* Hotfix: Regression in `LIFExodus`

## [v2.5] -- 2022-11-29

### Added

* Added support for Xylo-Audio v2 (SYNS61201) devices and HDK
* Added hardware versioning for Xylo devices
* Added a beta implementation of Quantisation-Aware Training for Torch backend in ``rockpool.transform.torch_transform``
* Added support for parameter boundary constraints in ``rockpool.training.torch_loss``
* Added tutorial for Spiking Heidelberg Digits audio classification
* Added tutorial and documentation for WaveSense network architecture
* Added support to ``LIFTorch`` for training decays and bitshift parameters
* Added a new utility package ``rockpool.utilities.tree_utils``

### Changed

* Updated support for Exodus v1.1
* Updated ``XyloSim.from_specification`` to handle  NIEN ≠ NRSN ≠ NOEN for Xylo devices
* Updated ``LIFTorch`` to provide proper ``tau``s for ``.as_graph()`` in case of decay and bitshift traning
* Improved backend management, to test torch version requirements

### Fixed

* Fixed usage of Jax optimisers in tutorial notebooks to reflect Jax API changes
* Fixed issues with ``LIFTorch`` and ``aLIFTorch``, preventing ``deepcopy`` protocol
* Fixed bug in `tree_utils`, where `Tree` was used instead of `dict` in `isinstance` check
* Replaced outdated reference from ``FFRateEuler`` to ``Rate`` module in high-level API tutorial
* Fixed seeds in torch and numpy to avoid ``nan`` loss problem while training in tutorial
* Fixed bug in ``TorchModule`` where assigning to an existing registered attribute would clear the family of the attribute
* Fixed a bug in Constant handling for `torch.Tensor`s, which would raise errors in torch 1.12
* Fixed bug in ``LIFTorch``, which would cause recorded state to hang around post-evolution, causing errors from `deepcopy`
* Fixed bug in `Module._register_module()`, where replacing an existing submodule would cause the string representation to be incorrect

## [v2.4.2] -- 2022-10-06

### Hotfix

Improved handling of weights when using `XyloSim.from_specification`

## [v2.4] -- 2022-08

### Major changes

- `Linear...` modules now *do not* have a bias parameter, by default.

### Added

- Support for Xylo SNN core v2, via XyloSim. Including biases and quantisation support; mapping and deployment for Xylo SNN core v2 (SYNS61201)
- Added support for Xylo-A2 test board, with audio recording support from Xylo AFE (`AFESamna` and `XyloSamna`)
- Support for an LIF neuron including a trainable adaptive threshold (`aLIFTorch`). Deployable to Xylo
- New module `BooleanState`, which maintains a boolean state
- Support for membrane potential training using `LIFExodus`

### Changed

- Xylo package support for HW versioning (SYNS61300; SYNS61201)
- Ability to return events, membrane potentials or synaptic currents as output from `XyloSim` and `XyloSamna`
- Enhanced Xylo `mapper` to be more lenient about weight matrix size --- now assumes missing weights are zero
- Xylo `mapper` is now more lenient about HW constraints, permitting larger numbers of input and output channels than supported by existing HDKs
- Xylo `mapper` supports a configurable number of maxmimum hidden and output neurons
- Running `black` is enforced by the CI pipeline
- `Linear...` modules now export bias parameters, if they are present
- `Linear...` modules now do not include bias parameters by default
- Xylo `mapper` now raises a warning if any linear weights have biases
- `LIFSlayer` renamed to `LIFExodus`, corresponding to `sinabs.exodus` library name change
- Periodic exponetial surrogate function now supports training thresholds

### Fixed

- Fixes related to torch modules moved to simulation devices
- Fixed issue in `dropout.py`, where if jax was missing an ImportError was raised
- Fixed an issue with `Constant` `torch` parameters, where `deepcopy` would raise an error
- Fixed issue with newer versions of torch; torch v1.12 is now supported
- Updated to support changes in latest jax api
- Fixed bug in `WavesenseNet`, where neuron class would not be checked properly
- Fixed bug in `channel_quantize`, where *un*quantized weights were returned instead of quantized weights

### Deprecated

- `LIFSlayer` is now deprecated

## [v2.3.1] -- 2022-03-24

### Hotfix

- Improved CI pipeline such that pipline is not blocked with sinabs.exodus cannot be installed
- Fixed UserWarning raised by some torch-backed modules
- Improved some unit tests

## [v2.3] -- 2022-03-16

### Added

- Standard dynamics introduced for LIF, Rate, Linear, Instant, ExpSyn. These are standardised across Jax, Torch and Numpy backends. We make efforts to guarantee identical dynamics for the standard modules across these backends, down to numerical precision
- LIF modules can now train threhsolds and biases as well as time constants
- New `JaxODELIF` module, which implements a trainable LIF neuron following common dynamical equations for LIF neurons
- New addition of the WaveSense network architecture, for temporal signal processing with SNNs. This is available in `rockpool.networks`, and is documented with a tutorial
- A new system for managing computational graphs, and mapping these graphs onto hardware architectures was introduced. These are documented in the Xylo quick-start tutorial, and in more detail in tutorials covering Computational Graphs and Graph Mapping. The mapping system performs design-rule checks for Xylo HDK
- Included methods for post-traning quantisation for Xylo, in `rockpool.transform`
- Added simulation of a divisive normalisation block for Xylo audio applications
- Added a `Residual` combinator, for convenient generation of networks with residual blocks
- Support for `sinabs` layers and Exodus
- `Module`, `JaxModule` and `TorchModule` provide facility for auto-batching of input data. Input data shape is `(B, T, Nin)`, or `(T, Nin)` when only a single batch is provided
- Expanded documentation on parameters and type-hinting

### Changed

- Python > 3.6 is now required
- Improved import handling, when various computational back-ends are missing
- Updated for new versions of `samna`
- Renamed Cimulator -> XyloSim
- Better parameter handling and rockpool/torch parameter registration for Torch modules
- (Most) modules can accept batched input data
- Improved / additional documentation for Xylo

### Fixed

- Improved type casting and device handling for Torch modules
- Fixed bug in Module, where `modules()` would return a non-ordered dict. This caused issues with `JaxModule`

### Removed

- Removed several obsolete `Layer`s and `Network`s from Rockpool v1

## [v2.2] -- 2021-09-09

### Added

- Added support for the Xylo development kit in `.devices.xylo`, including several tutorials
- Added CTC loss implementations in `.training.ctc_loss`
- New trainable `torch` modules: `LIFTorch` and others in `.nn.modules.torch`, including an asynchronous delta modulator `UpDownTorch`
- Added `torch` training utilities and loss functions in `.training.torch_loss`
- New `TorchSequential` class to support `Sequential` combinator for `torch` modules
- Added a `FFwdStackTorch` class to support `FFwdStack` combinator for `torch` modules

### Changed

- Existing `LIFTorch` module renamed to `LIFBitshiftTorch`; updated module to align better with Rockpool API
- Improvements to `.typehints` package
- `TorchModule` now raises an error if submodules are not `Torchmodules`

### Fixed

- Updated LIF torch training tutorial to use new `LIFBitshiftTorch` module
- Improved installation instructions for `zsh`

## [v2.1] -- 2021-07-20

### Added

- 👹 Adversarial training of parameters using the *Jax* back-end, including a tutorial
- 🐰 "Easter" tutorial demonstrating an SNN trained to generate images
- 🔥 Torch tutorials for training non-spiking and spiking networks with Torch back-ends
- Added new method `nn.Module.timed()`, to automatically convert a module to a `TimedModule`
- New `LIFTorch` module that permits training of neuron and synaptic time constants in addition to other network parameters
- New `ExpSynTorch` module: exponential leak synapses with Torch back-end
- New `LinearTorch` module: linear model with Torch back-end
- New `LowPass` module: exponential smoothing with Torch back-end
- New `ExpSmoothJax` module: single time-constant exponential smoothing layer, supporting arbitrary transfer functions on output
- New `softmax` and `log_softmax` losses in `jax_loss` package
- New `utilities.jax_tree_utils` package containing useful parameter tree handling functions
- New `TSContinuous.to_clocked()` convenience method, to easily rasterise a continuous time series
- Alpha: Optional `_wrap_recorded_state()` method added to `nn.Module` base class, which supports wrapping recorded state dictionaries as `TimeSeries` objects, when using the high-level `TimeSeries` API
- Support for `add_events` flag for time-series wrapper class
- New Parameter dictionary classes to simplify conversion and handling of *Torch* and *Jax* module parameters
  - Added `astorch()` method to parameter dictionaries returned form `TorchModule`
- Improved type hinting

### Changed

- Old `LIFTorch` module renamed to `LIFBitshiftTorch`
- Kaiming and Xavier initialisation support for `Linear` modules
- `Linear` modules provide a bias by default
- Moved `filter_bank` package from V1 layers into `nn.modules`
- Update *Jax* requirement to > v2.13

### Fixed

- Fixed *binder* links for tutorial notebooks
- Fixed bug in `Module` for multiple inheritance, where the incorrect `__repr__()` method would be called
- Fixed `TimedModuleWrapper.reset_state()` method
- Fixed axis limit bug in `TSEvent.plot()` method
- Removed page width constraint for docs
- Enable `FFExpSyn` module by making it independent of old `RRTrainedLayer`

### Deprecated

- Removed `rpyc` dependency

### Removed

## [v2.0] -- 2021-03-24

- **New Rockpool API. Breaking change from v1.x**
- Documentation for new API
- Native support for Jax and Torch backends
- Many v1 Layers transferred

## [v1.1.0.4] -- 2020-11-06

- Hotfix to remove references to ctxctl and aiCTX
- Hotfix to include NEST documentation in CI-built docs
- Hotfix to include change log in build docs

## [v1.1] -- 2020-09-12

### Added

- Considerably expanded support for Denève-Machens spike-timing networks, including training arbitrary dynamical systems in a new `RecFSSpikeADS` layer. Added tutorials for standard D-M networks for linear dynamical systems, as well as a tutorial for training ADS networks
- Added a new "Intro to SNNs" getting-started guide
- A new "sharp points of Rockpool" tutorial collects the tricks and traps for new users and old
- A new `Network` class, `JaxStack`, supports stacking and end-to-end gradient-based training of all Jax-based layers. A new tutorial has been added for this functionality
- `TimeSeries` classes now support best-practices creation from clock or rasterised data. `TSContinuous` provides a `.from_clocked()` method, and `TSEvent` provides a `.from_raster()` method for this purpose. `.from_clocked()` a sample-and-hold interpolation, for intuitive generation of time series from periodically-sampled data.
- `TSContinuous` now supports a `.fill_value` property, which permits extrapolation using `scipy.interpolate`
- New `TSDictOnDisk` class for storing `TimeSeries` objects transparently on disk
- Allow ignoring data points for specific readout units in ridge regression Fisher relabelling. To be used, for example with all-vs-all classification
- Added exponential synapse Jax layers
- Added `RecLIFCurrentIn_SO` layer

### Changed

- `TSEvent` time series no longer support creation without explicitly setting `t_stop`. The previous default of taking the final event time as `t_stop` was causing too much confusion. For related reasons, `TSEvent` now forbids events to occur at `t_stop`
- `TimeSeries` classes by default no longer permit sampling outside of the time range they are defined for, raising a `ValueError` exception if this occurs. This renders safe several traps that new users were falling in to. This behaviour is selectable per time series, and can be transferred to a warning instead of an exception using the `beyond_range_exception` flag
- Jax trainable layers now import from a new mixin class `JaxTrainer`. THe class provides a default loss function, which can be overridden in each sub-class to provide suitable regularisation. The training interface now returns loss value and gradients directly, rather than requiring an extra function call and additional evolution
- Improved training method for JAX rate layers, to permit parameterisation of loss function and optimiser
- Improved the `._prepare_input...()` methods in the `Layer` class, such that all `Layer`s that inherit from this superclass are consistent in the number of time steps returned from evolution
- The `Network.load()` method is now a class method
- Test suite now uses multiple cores for faster testing
- Changed company branding from aiCTX -> SynSense
- Documentation is now hosted at [https://rockpool.ai](https://rockpool.ai)

### Fixed

- Fixed bugs in precise spike-timing layer `RecSpikeBT`
- Fixed behavior of `Layer` class when passing weights in wrong format
- Stability improvements in `DynapseControl`
- Fix faulty z_score_standardization and Fisher relabelling in `RidgeRegrTrainer`. Fisher relabelling now has better handling of differently sized batches
- Fixed bugs in saving and loading several layers
- More sensible default values for `VirtualDynapse` baseweights
- Fix handling of empty `channels` argument in `TSEvent._matching_channels()` method
- Fixed bug in `Layer._prepare_input`, where it would raise an AssertionError when no input TS was provided
- Fixed a bug in `train_output_target`, where the gradient would be incorrectly handled if no batching was performed
- Fixed `to_dict` method for `FFExpSynJax` classes
- Removed redundant `_prepare_input()` method from Torch layer
- Many small documentation improvements

---

## [v1.0.8] -- 2020-01-17

### Added

- Introduced new `TimeSeries` class method `concatenate_t()`, which permits construction of a new time series by concatenating a set of existing time series, in the time dimension
- `Network` class now provides a `to_dict()` method for export. `Network` now also can treat sub-`Network`s as layers.
- Training methods for spiking LIF Jax-backed layers in `rockpool.layers.training`. Tutorial demonstrating SGD training of a feed-forward LIF network. Improvements in JAX LIF layers.
- Added `filter_bank` layers, providing `layer` subclasses which act as filter banks with spike-based output
- Added a `filter_width` parameter for butterworth filters
- Added a convenience function `start_at_zero()` to delay `TimeSeries` so that it starts at 0
- Added a change log in `CHANGELOG.md`

### Changed

- Improved `TSEvent.raster()` to make it more intuitive. Rasters are now produced in line with time bases that can be created easily with `numpy.arange()`
- Updated `conda_merge_request.sh` to work for conda feedstock
- `TimeSeries.concatenate()` renamed to `concatenate_t()`
- `RecRateEuler` warns if `tau` is too small instead of silently changing `dt`

### Fixed or improved

- Fixed issue in `Layer`, where internal property was used when accessing `._dt`. This causes issues with layers that have an unusual internal type for `._dt` (e.g. if data is stored in a JAX variable on GPU)
- Reduce memory footprint of `.TSContinuous` by approximately half
- Reverted regression in layer class `.RecLIFJax_IO`, where `dt` was by default set to `1.0`, instead of being determined by `tau_...`
- Fixed incorrect use of `Optional[]` type hints
- Allow for small numerical differences in comparison between weights in NEST test `test_setWeightsRec`
- Improvements in inline documentation
- Increasing memory efficiency of `FFExpSyn._filter_data` by reducing kernel size
- Implemented numerically stable timestep count for TSEvent rasterisation
- Fixed bugs in `RidgeRegrTrainer`
- Fix plotting issue in time series
- Fix bug of RecRateEuler not handling `dt` argument in `__init__()`
- Fixed scaling between torch and nest weight parameters
- Move `contains()` method from `TSContinuous` to `TimeSeries` parent class
- Fix warning in `RRTrainedLayer._prepare_training_data()` when times of target and input are not aligned
- Brian layers: Replace `np.asscalar` with `float`

---

## [v1.0.7.post1] -- 2019-11-28

### Added

- New `.Layer` superclass `.RRTrainedLayer`. This superclass implements ridge regression for layers that support ridge regression training
- `.TimeSeries` subclasses now add axes labels on plotting
- New spiking LIF JAX layers, with documentation and tutorials `.RecLIFJax`, `.RecLIFJax_IO`, `.RecLIFCurrentInJax`, `.RecLIFCurrentInJAX_IO`
- Added `save` and `load` facilities to `.Network` objects
- `._matching_channels()` now accepts an arbitrary list of event channels, which is used when analysing a periodic time series

### Changed

- Documentation improvements
- :py:meth:`.TSContinuous.plot` method now supports ``stagger`` and ``skip`` arguments
- `.Layer` and `.Network` now deal with a `.Layer.size_out` attribute. This is used to determine whether two layers are compatible to connect, rather than using `.size`
- Extended unit test for periodic event time series to check non-periodic time series as well

### Fixed

- Fixed bug in `TSEvent.plot()`, where stop times were not correctly handled
- Fix bug in `Layer._prepare_input_events()`, where if only a duration was provided, the method would return an input raster with an incorrect number of time steps
- Fixed bugs in handling of periodic event time series `.TSEvent`
- Bug fix: `.Layer._prepare_input_events` was failing for `.Layer` s with spiking input
- `TSEvent.__call__()` now correctly handles periodic event time series

---

## [v1.0.6] -- 2019-11-01

- CI build and deployment improvements

---

## [v1.0.5] -- 2019-10-30

- CI Build and deployment improvements

---

## [v1.0.4] -- 2019-10-28

- Remove deployment dependency on docs
- Hotfix: Fix link to `Black`
- Add links to gitlab docs

---

## [v1.0.3] -- 2019-10-28

- Hotfix for incorrect license text
- Updated installation instructions
- Included some status indicators in readme and docs
- Improved CI
- Extra meta-data detail in `setup.py`
- Added more detail for contributing
- Update README.md

---

## [v1.0.2] -- 2019-10-25

- First public release
