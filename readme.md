This repository contains a set of functions to pre-process and process electroencephalography (EEG) data.

# Introduction

With most recording devices, EEG data are structured as a big matrix of shape (time x electrodes). One electrode channel generaly corresponds to the trigger channel used to synchronise the participant response or the stimuli to the EEG signal. The raw EEG can be split in chunks of time according to this trigger channel. It is then possible to average EEG signal coming from same condition for instance.

These functions can be used to load data, do some kind of processing, plot etc.

# Special functions

## Denoising source separation

This denoising method is an implementation of [this matlab toolbox]() created by Alain de Cheveigné. More details about this method can be found in the following papers:

-
-

# API

### `addOffset(data, offset)`

Plot all electrodes with an offset from t0 to t1. The stimulus channel is
also ploted and red lines are used to show the events.

- **`data`** `instance of pandas.core.DataFrame`

   Add offset to data.
- **`offset`** `float`

   Value of the offset.

Returns:

- **`newData`** `instance of pandas.core.DataFrame`

   The data with offset applied to each electrode.

### `calculateBaseline(data, baselineDur=0.1, fs=2048.)`

Calculate and return the baseline (average of each data point) of a signal.
The baseline will calculated from the first `baselineDur` seconds of this
signal.

- **`data`** `instance of pandas.core.DataFrame`

   Data used to calculate the baseline.
- **`baselineDur`** `float`

   Duration of the baseline to use for the calulation of the average in
seconds.
- **`fs`** `float`

   Sampling frequency of data in Hz.

Returns:

- **`baseline`** `float`

   The baseline value.

### `chebyBandpassFilter(data, cutoff, gstop=40, gpass=1, fs=2048.)`

Design a filter with scipy functions avoiding unstable results (when using
ab output and filtfilt(), lfilter()...).
Cf. ()[]

- **`data`** `instance of numpy.array | instance of pandas.core.DataFrame`

   Data to be filtered. Each column will be filtered if data is a
dataframe.
- **`cutoff`** `array-like of float`

   Pass and stop frequencies in order:
    - the first element is the stop limit in the lower bound
    - the second element is the lower bound of the pass-band
    - the third element is the upper bound of the pass-band
    - the fourth element is the stop limit in the upper bound
For instance, [0.9, 1, 45, 48] will create a band-pass filter between
1 Hz and 45 Hz.
- **`gstop`** `int`

   The minimum attenuation in the stopband (dB).
- **`gpass`** `int`

   The maximum loss in the passband (dB).

Returns:

- **`filteredData`** `instance of numpy.array | instance of pandas.core.DataFrame`

   The filtered data.

### `checkPlots(data1, data2, fs1, fs2, start, end, electrodeNum)`

Check filtering and downsampling by ploting both datasets.

- **`data1`** `instance of pandas.core.DataFrame`

   First dataframe.
- **`data2`** `instance of pandas.core.DataFrame`

   Second dataframe.
- **`fs1`** `float`

   Sampling frequency of the first dataframe in Hz.
- **`fs2`** `float`

   Sampling frequency of the second dataframe in Hz.
- **`start`** `float`

   Start of data to plot in seconds.
- **`end`** `float`

   End of data to plot in seconds.
- **`electrodeNum`** `int`

   Index of the column to plot.

Returns:

- **`fig`** `instance of matplotlib.figure.Figure`

   The figure containing both dataset plots.

### `checkPlotsNP(data1, data2, fs1, fs2, start, end, electrodeNum)`

Check filtering and downsampling by ploting both datasets.

- **`data1`** `instance of pandas.core.DataFrame`

   First dataframe.
- **`data2`** `instance of pandas.core.DataFrame`

   Second dataframe.
- **`fs1`** `float`

   Sampling frequency of the first dataframe in Hz.
- **`fs2`** `float`

   Sampling frequency of the second dataframe in Hz.
- **`start`** `float`

   Start of data to plot in seconds.
- **`end`** `float`

   End of data to plot in seconds.
- **`electrodeNum`** `int`

   Index of the column to plot.

Returns:

- **`fig`** `instance of matplotlib.figure.Figure`

   The figure containing both dataset plots.

### `computeFFT(data, fs)`

Compute the FFT of `data` and return also the axis in Hz for further plot.

- **`data`** `array`

   First dataframe.
- **`fs`** `float`

   Sampling frequency in Hz.

Returns:

- **`fAx`** `instance of numpy.array`

   Axis in Hz to plot the FFT.
- **`fftData`** `instance of numpy.array`

   Value of the fft.

### `computePickEnergy(data, pickFreq, showPlot, fs)`

Calculate the relative energy at the frequency `pickFreq` from the the FFT
of `data`. Compare the mean around the pick with the mean of a broader zone
for each column.

- **`data`** `array-like`

   Matrix of the shape (time, electrode).
- **`pickFreq`** `float`

   Frequency in Hz of the pick for which we want to calculate the relative energy.
- **`showPlot`** `boolean`

   A plot of the FFT can be shown.
- **`fs`** `float`

   Sampling frequency in Hz.

Returns:

- **`pickRatio`** `float`

   Relative energy of the pick.

### `create3DMatrix(data, trialTable, events, trialList, fs)`


### `createStimChannel(events)`

Create stim channel from events.

- **`events`** `instance of pandas.core.DataFrame`

   Dataframe containing list of events obtained with mne.find_events(raw)
   .

Returns:

- **`stim`** `instance of pandas.core.series.Series`

   Series containing the stimulus channel reconstructed from events.

### `discriminateEvents(events, threshold)`

Discriminate triggers when different kind of events are on the same channel.
A time threshold is used to determine if two events are from the same trial.

- **`events`** `instance of pandas.core.DataFrame`

   Dataframe containing the list of events obtained with
mne.find_events(raw).
- **`threshold`** `float`

   Time threshold in milliseconds. Keeps an event if the time difference
with the next one is superior than threshold.

Returns:

- **`newData`** `instance of pandas.series.Series`

   List of trial number filling the requirements.

### `downsample(data, oldFS, newFS)`

Resample data from oldFS to newFS using the scipy 'resample' function.

- **`data`** `instance of pandas.core.DataFrame`

   Data to resample.
- **`oldFS`** `float`

   The sampling frequency of data.
- **`newFS`** `float`

   The new sampling frequency.

Returns:

- **`newData`** `instance of pandas.DataFrame`

   The downsampled dataset.

### `downsampleEvents(events, oldFS, newFS)`

Modify the timestamps of events to match a new sampling frequency.

- **`events`** `instance of pandas.core.DataFrame`

   Dataframe containing list of events obtained with mne.find_events(raw)
   .
- **`oldFS`** `float`

   The sampling frequency of the input events.
- **`newFS`** `float`

   The sampling frequency to the output events.

Returns:

- **`newEvents`** `instance of pandas.DataFrame`

   DataFrame containing the downsampled events.

### `downsampleNP(data, oldFS, newFS)`

Resample data from oldFS to newFS using the scipy 'resample' function.

- **`data`** `instance of pandas.core.DataFrame`

   Data to resample.
- **`oldFS`** `float`

   The sampling frequency of data.
- **`newFS`** `float`

   The new sampling frequency.

Returns:

- **`newData`** `instance of pandas.DataFrame`

   The downsampled dataset.

### `FFTTrials(data, events, trialNumList, baselineDur, trialDur, fs, normalize`


### `getBehaviorData(dbAddress, dbName, sessionNum)`

Fetch behavior data from couchdb (SOA, SNR and trial duration).

- **`dbAddress`** `str`

   Path to the couch database.
- **`dbName`** `str`

   Name of the database on the couch instance.
- **`sessionNum`** `int`

   Behavior data will be fetched from this sessionNum.

Returns:

- **`lookupTable`** `instance of pandas.core.DataFrame`

   A dataframe containing trial data.

### `getEvents(raw, eventCode)`

Get the events corresponding to `eventCode`.

- **`raw`** `instance of mne.io.edf.edf.RawEDF`

   RawEDF object from the MNE library containing data from the .bdf files.
- **`eventCode`** `int`

   Code corresponding to a specific events. For instance, with a biosemi
device, the triggers are coded 65284, 65288 and 65296 respectively on
the first, second and third channel.

Returns:

- **`startEvents`** `instance of pandas.core.DataFrame`

   Dataframe containing the list of timing corresponding to the event code
in the first column. The second column contains the code before the event
and the third the code of the selected event.

### `getTrialsAverage(data, events, trialDur=None, trialNumList=None`


### `getTrialData(data, events, trialNum=0, electrode=None, baselineDur=0.1`


### `getTrialDataNP(data, events, trialNum=0, electrode=None, baselineDur=0.1`


### `getTrialNumList(table, **kwargs)`

Returns a subset of table according to SOA, SNR and/or targetFreq. This is
used to select trials with specific parameters.

- **`table`** `instance of pandas.core.DataFrame`

   DataFrame containing trial number and their parameters (SOA, SNR...).
- **`kwargs`** `array-like of int | None`

   Array containing element from table to select. It can be `SOA`, `SNR` or
`targetFreq`.

Returns:

- **`newData`** `instance of pandas.series.Series`

   List of trial number filling the requirements.

### `importH5(name, df)`


### `loadEEG(path)`

Load data from .bdf files. If an array of path is provided, files will be
concatenated.

- **`path`** `str | array-like of str`

   Path to the .bdf file(s) to load.

Returns:

- **`raw`** `instance of mne.io.edf.edf.RawEDF`

   RawEDF object from the MNE library containing data from the .bdf files.

### `normalizeFromBaseline(data, baselineDur=0.1, fs=2048.)`

Normalize data by subtracting the baseline to each data point. The data used
to normalize has to be included at the beginning of data. For instance, to
normalize a 10 seconds signal with a 0.1 second baseline, data has to be
10.1 seconds and the baseline used will be the first 0.1 second.

- **`data`** `instance of pandas.core.DataFrame`

   Data to normalize.
- **`baselineDur`** `float`

   Duration of the baseline to use for the normalization in seconds.
- **`fs`** `float`

   Sampling frequency of data in Hz.

Returns:

- **`normalized`** `instance of pandas.core.DataFrame`

   The normalized data.

### `plot3DMatrix(data, picks, trialList, average, fs)`


### `plotDataSubset(data, stim, events, offset, t0=0, t1=1, fs=2048.)`

Plot all electrodes with an offset from t0 to t1. The stimulus channel is
also ploted and red lines are used to show the events.

- **`data`** `instance of pandas.core.DataFrame`

   Data to plot (not epoched). Columns correspond to electrodes.
- **`stim`** `instance of pandas.core.DataFrame`

   One column dataframe containing the event codes. Used to plot the
stimulus timing along with EEG.
- **`events`** `instance of pandas.core.DataFrame`

   Dataframe containing the list of events obtained with
mne.find_events(raw).
- **`offset`** `float`

   Offset between each electrode line on the plot.
- **`t0`** `float`

   Start of data to plot.
- **`t1`** `float`

   End of data to plot.
- **`fs`** `float`

   Sampling frequency of data in Hz.

Returns:

- **`fig`** `instance of matplotlib.figure.Figure`

   The figure of the data subset in the time domain.

### `plotERPElectrodes(data, trialNumList, events, trialDur=None, fs=2048.`

startOffset=0):

### `plotFFT(data, facet=False, freqMin=None, freqMax=None, yMin=None`


### `plotFFTElectrodes(data, trialNumList, events, trialDur, fs`

freqMin=None, freqMax=None, yMin=None, yMax=None, startOffset=0, noiseAve=None):

### `plotFFTNP(data, average, fs)`


### `plotFilterResponse(zpk, fs)`

Plot the filter frequency response.

- **`zpk`** `array-like`

   The 3 parameters of the filter [z, p, k].
- **`fs`** `float`

   Sampling frequency in Hz.

Returns:

- **`fig`** `instance of matplotlib.figure.Figure`

   The figure of the filter response.

### `refToAverageNP(data)`


### `refToMastoids(data, M1, M2)`

Transform each electrode of data according to the average of M1 and M2.

- **`data`** `instance of pandas.core.DataFrame`

   First column has to contain the timing of events in frames.
- **`M1`** `instance of pandas.core.series.Series`

   Values of mastoid 1. This Series has to be the same length as data.
- **`M2`** `instance of pandas.core.series.Series`

   Values of mastoid 2. This Series has to be the same length as data

Returns:

- **`newData`** `instance of pandas.core.DataFrame`

   A dataframe referenced to matoids containing all electrode from which
we subtract the average of M1 and M2.

### `refToMastoidsNP(data, M1, M2)`


### `compareTimeBehaviorEEG(dbAddress, dbName, events, startSound, interTrialDur`


### `preprocessing(files)`


### `getBehaviorTables(dbAddress, dbName)`


### `mergeBehaviorTables(tableHJ1, tableHJ2, tableHJ3)`



# Requirements

It uses some methods of the [MNE library]() and heavily depends on pandas and numpy.