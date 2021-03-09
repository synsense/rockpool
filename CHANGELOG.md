# Change log

All notable changes between Rockpool releases will be documented in this file.

## [v2.0] -- 2021-03-09

 - New Rockpool API
 - Breaking change from v1.x

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
 - Documentation is now hosted at [https://rockpool.ai]()
 
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

-  Hotfix for incorrect license text
-  Updated installation instructions
-  Included some status indicators in readme and docs
- Improved CI
-  Extra meta-data detail in `setup.py`
-  Added more detail for contributing
-  Update README.md

---
## [v1.0.2] -- 2019-10-25

- First public release
