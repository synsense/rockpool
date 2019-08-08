.. _timeseriesdocs:

Working with time series data
=============================

Concepts
--------
In |project|, temporal data ("time series" data) is encapsulated in a set of classes that derive from :py:class:`.TimeSeries`. Time series come in two basic flavours: "continuous" time series, which have been sampled at some set of time points but which represent values that can exist at any point in time; and "event" time series, which consist of discrete event times.

The :py:class:`.TimeSeries` subclasses provide methods for extracting, resampling, shifting, trimming and manipulating time series data in a convenient fashion. Since |project| naturally deals with temporal dynamics and temporal data, :py:class:`.TimeSeries` objects are used to pass around time series data both as input and as output.

Continuous time series
----------------------

Continuous time series consist of a list of sample times in increasing temporal order (the "time base" of the time series), and a corresponding array of sampled values. A single time series object can contain several channel's worth of data, which have been sampled on a common time base. Continuous time series are represented by the :py:class:`.TSContinuous` class.

Building time series
********************

To build a time series, you need at minimum a time base and a list of samples: ::

    import numpy as np
    from |project| import TSContinuous

    times = np.linspace(0, 2, 100)
    samples = np.sin(times * np.pi)
    series = TSContinuous(times, samples)


``series`` is now a :py:class:`.TSContinuous` object containing the time base and samples. You can examine it by plotting, either with ``matplotlib`` or with ``HolowViews``:

``series.plot()``

However, you can also do so much more! For example, you can easily scale the time series values:

``(series * 2).plot()``

You can shift the sample values:

``(series + 1).plot()``

You can perform

You can concatenate two time series:

``series.concatenate_t(series).plot()``

And much more: see other available methods in :py:class:`.TSContinuous`.


Resampling and interpolating time series
****************************************

Since these time series represent continuous-time data, they can be sampled


Periodic time series
********************


Event-based time series
-----------------------


Loading time series from a file
-------------------------------

.. autofunction::
    timeseries.load_ts_from_file


Managing the plotting backend
-----------------------------

.. autofunction::
    timeseries.set_global_plotting_backend
    timeseries.get_global_plotting_backend
