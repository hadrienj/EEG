#### `addOffset(data, offset)`

Plot all electrodes with an offset from t0 to t1. The stimulus channel is
also ploted and red lines are used to show the events.

- **`data`** `instance of pandas.core.DataFrame`

      Add offset to data.
- **`offset`** `float`

   Value of the offset.

Returns
-------
   - **`newData`** `instance of pandas.core.DataFrame`

The data with offset applied to each electrode.

#### `calculateBaseline(data, baselineDur=0.1, fs=2048.)`

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

Returns
-------
   - **`baseline`** `float`

The baseline value.

#### `chebyBandpassFilter(data, cutoff, gstop=40, gpass=1, fs=2048.)`

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

Returns
-------
   - **`filteredData`** `instance of numpy.array | instance of pandas.core.DataFrame`

The filtered data.

#### `checkPlots(data1, data2, fs1, fs2, start, end, electrodeNum)`

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

Returns
-------
   - **`fig`** `instance of matplotlib.figure.Figure`

The figure containing both dataset plots.

#### `checkPlotsNP(data1, data2, fs1, fs2, start, end, electrodeNum)`

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

Returns
-------
   - **`fig`** `instance of matplotlib.figure.Figure`

The figure containing both dataset plots.

#### `compareTimeBehaviorEEG(dbAddress, dbName, events, startSound, interTrialDur`


#### `computeFFT(data, fs)`

Compute the FFT of `data` and return also the axis in Hz for further plot.

- **`data`** `array`

   First dataframe.
- **`fs`** `float`

   Sampling frequency in Hz.

Returns
-------
   - **`fAx`** `instance of numpy.array`

Axis in Hz to plot the FFT.
- **`fftData`** `instance of numpy.array`

   Value of the fft.

#### `computePickEnergy(data, pickFreq, showPlot, fs)`

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

Returns
-------
   - **`pickRatio`** `float`

Relative energy of the pick.

#### `create3DMatrix(data, trialTable, events, trialList, fs)`

trials = trials[trials['trialNum'].isin(trialList)]
totalTrialNum = np.max(trials['trialNum'])
m = trials.shape[0]
print m, totalTrialNum

electrodeNumber = data.shape[1]
trialDur = 10
baselineDur = 0
trialSamples = int(np.round((trialDur+baselineDur)*fs))
# number of features: each sample for each electrode
n = int(np.round(trialDur*fs*electrodeNumber))
# Get trial data
X = np.zeros((m, trialSamples, electrodeNumber))

print 'creating matrix of shape (trials=%d, time=%d, electrodes=%d)' % (X.shape[0],
                                                                    X.shape[1],
                                                                    X.shape[2])
count = 0
for i in range(totalTrialNum+1):
# Check if this trial is in our subset
if (i in trialList.unique()):
    trial = getTrialDataNP(data.values, events=events,
                           trialNum=i, baselineDur=baselineDur,
                           startOffset=0,
                           trialDur=trialDur, fs=fs)
    X[count, :, :] = trial
    count += 1
print X.shape
return X

createStimChannel(events):

#### `createStimChannel(events)`

Create stim channel from events.

- **`events`** `instance of pandas.core.DataFrame`

   Dataframe containing list of events obtained with mne.find_events(raw)
   .

Returns
-------
   - **`stim`** `instance of pandas.core.series.Series`

Series containing the stimulus channel reconstructed from events.

#### `discriminateEvents(events, threshold)`

Discriminate triggers when different kind of events are on the same channel.
A time threshold is used to determine if two events are from the same trial.

- **`events`** `instance of pandas.core.DataFrame`

   Dataframe containing the list of events obtained with
mne.find_events(raw).
- **`threshold`** `float`

   Time threshold in milliseconds. Keeps an event if the time difference
with the next one is superior than threshold.

Returns
-------
   - **`newData`** `instance of pandas.series.Series`

List of trial number filling the requirements.

#### `downsample(data, oldFS, newFS)`

Resample data from oldFS to newFS using the scipy 'resample' function.

- **`data`** `instance of pandas.core.DataFrame`

   Data to resample.
- **`oldFS`** `float`

   The sampling frequency of data.
- **`newFS`** `float`

   The new sampling frequency.

Returns
-------
   - **`newData`** `instance of pandas.DataFrame`

The downsampled dataset.

#### `downsampleEvents(events, oldFS, newFS)`

Modify the timestamps of events to match a new sampling frequency.

- **`events`** `instance of pandas.core.DataFrame`

   Dataframe containing list of events obtained with mne.find_events(raw)
   .
- **`oldFS`** `float`

   The sampling frequency of the input events.
- **`newFS`** `float`

   The sampling frequency to the output events.

Returns
-------
   - **`newEvents`** `instance of pandas.DataFrame`

DataFrame containing the downsampled events.

#### `downsampleNP(data, oldFS, newFS)`

Resample data from oldFS to newFS using the scipy 'resample' function.

- **`data`** `instance of pandas.core.DataFrame`

   Data to resample.
- **`oldFS`** `float`

   The sampling frequency of data.
- **`newFS`** `float`

   The new sampling frequency.

Returns
-------
   - **`newData`** `instance of pandas.DataFrame`

The downsampled dataset.

#### `FFTTrials(data, events, trialNumList, baselineDur, trialDur, fs, normalize`

dataElectrodes = np.zeros((5171, len(electrodes)))
countEle = 0
for electrode in electrodes:
print 'electrode number %d' %electrode
allTrials = np.zeros((5171, len(trialNumList)))
count = 0
for trialNum in trialNumList:
    trialData = getTrialDataNP(data, events=events, trialNum=trialNum, electrode=electrode,
            baselineDur=baselineDur, trialDur=trialDur, fs=fs)
    if normalize:
        trialData = normalizeFromBaseline(trialData,
            baselineDur=baselineDur, fs=fs)
    Y = fftpack.fft(trialData)
    allTrials[:, count] = pd.Series(Y.real)
    count += 1
dataElectrodes[:, countEle] = allTrials.mean(axis=1)
countEle += 1
return dataElectrodes

getBehaviorData(dbAddress, dbName, sessionNum):

#### `getBehaviorData(dbAddress, dbName, sessionNum)`

Fetch behavior data from couchdb (SOA, SNR and trial duration).

- **`dbAddress`** `str`

   Path to the couch database.
- **`dbName`** `str`

   Name of the database on the couch instance.
- **`sessionNum`** `int`

   Behavior data will be fetched from this sessionNum.

Returns
-------
   - **`lookupTable`** `instance of pandas.core.DataFrame`

A dataframe containing trial data.

#### `getBehaviorTables(dbAddress, dbName)`


# tableHJ2 is in split in two sessions
tableHJ2 = getBehaviorData(dbAddress, dbName, sessionNum=144)
tableHJ2_secondSession = getBehaviorData(dbAddress, dbName, sessionNum=145)
tableHJ2_secondSession['trialNum'] += 81
tableHJ2 = tableHJ2.append(tableHJ2_secondSession)

tableHJ3 = getBehaviorData(dbAddress, dbName, sessionNum=147)
return tableHJ1, tableHJ2, tableHJ3

mergeBehaviorTables(tableHJ1, tableHJ2, tableHJ3):
tableHJ1Temp = tableHJ1.copy()
tableHJ1Temp['session'] = 1
# Last trial of first session: trigger but no answer so no behavior response
# Add a fake behavioral trial
lenHJ1 = tableHJ1Temp.shape[0]
tableHJ1Temp.loc[lenHJ1+1] = [lenHJ1, 0, 0, 0, 1, 0]

tableHJ2Temp = tableHJ2.copy()
tableHJ2Temp['trialNum'] += lenHJ1 + 1
tableHJ2Temp['session'] = 2

tableHJ3Temp = tableHJ3.copy()
tableHJ3Temp['trialNum'] += lenHJ1 + 1 + tableHJ2.shape[0] 
tableHJ3Temp['session'] = 3

tableAll = tableHJ1Temp.append([tableHJ2Temp, tableHJ3Temp], ignore_index=True)
tableAll = tableAll.sort_values('trialNum')
tableAll = tableAll.reset_index(drop=True)

return tableAll


#### `getEvents(raw, eventCode)`

Get the events corresponding to `eventCode`.

- **`raw`** `instance of mne.io.edf.edf.RawEDF`

   RawEDF object from the MNE library containing data from the .bdf files.
- **`eventCode`** `int`

   Code corresponding to a specific events. For instance, with a biosemi
device, the triggers are coded 65284, 65288 and 65296 respectively on
the first, second and third channel.

Returns
-------
   - **`startEvents`** `instance of pandas.core.DataFrame`

Dataframe containing the list of timing corresponding to the event code
in the first column. The second column contains the code before the event
and the third the code of the selected event.

#### `getTrialData(data, events, trialNum=0, electrode=None, baselineDur=0.1`


#### `getTrialDataNP(data, events, trialNum=0, electrode=None, baselineDur=0.1`

# See getTrialData
baselineDurSamples = int(np.round(baselineDur * fs))
startOffsetSamples = int(np.round(startOffset * fs))
start = events[0][trialNum]-baselineDurSamples+startOffsetSamples
if (trialDur is None):
if (trialNum<events[0].shape[0]-1):
    end = events[0][trialNum+1]
else:
    # arbitrarly set trial duration to 10s for last trial
    lastTrialDur = int(np.round(10 * fs))
    end = events[0][trialNum]+lastTrialDur
else:
durSamples = int(np.round(trialDur * fs))
end = events[0][trialNum]+durSamples

if (electrode is None):
dataElectrode = data[start:end, :]
else:
dataElectrode = data[start:end, electrode]

return dataElectrode

getTrialNumList(table, **kwargs):

#### `getTrialNumList(table, **kwargs)`

Returns a subset of table according to SOA, SNR and/or targetFreq. This is
used to select trials with specific parameters.

- **`table`** `instance of pandas.core.DataFrame`

   DataFrame containing trial number and their parameters (SOA, SNR...).
- **`kwargs`** `array-like of int | None`

   Array containing element from table to select. It can be `SOA`, `SNR` or
`targetFreq`.

Returns
-------
   - **`newData`** `instance of pandas.series.Series`

List of trial number filling the requirements.

#### `getTrialsAverage(data, events, trialDur=None, trialNumList=None`


#### `importH5(name, df)`

data = f.get(df)

data = np.array(data)
oldShape = data.shape
data = np.swapaxes(data, 1, 2)
print 'convert shape %s to %s' % (oldShape, data.shape)
return data

loadEEG(path):

#### `loadEEG(path)`

Load data from .bdf files. If an array of path is provided, files will be
concatenated.

- **`path`** `str | array-like of str`

   Path to the .bdf file(s) to load.

Returns
-------
   - **`raw`** `instance of mne.io.edf.edf.RawEDF`

RawEDF object from the MNE library containing data from the .bdf files.

#### `mergeBehaviorTables(tableHJ1, tableHJ2, tableHJ3)`

tableHJ1Temp['session'] = 1
# Last trial of first session: trigger but no answer so no behavior response
# Add a fake behavioral trial
lenHJ1 = tableHJ1Temp.shape[0]
tableHJ1Temp.loc[lenHJ1+1] = [lenHJ1, 0, 0, 0, 1, 0]

tableHJ2Temp = tableHJ2.copy()
tableHJ2Temp['trialNum'] += lenHJ1 + 1
tableHJ2Temp['session'] = 2

tableHJ3Temp = tableHJ3.copy()
tableHJ3Temp['trialNum'] += lenHJ1 + 1 + tableHJ2.shape[0] 
tableHJ3Temp['session'] = 3

tableAll = tableHJ1Temp.append([tableHJ2Temp, tableHJ3Temp], ignore_index=True)
tableAll = tableAll.sort_values('trialNum')
tableAll = tableAll.reset_index(drop=True)

return tableAll


#### `normalizeFromBaseline(data, baselineDur=0.1, fs=2048.)`

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

Returns
-------
   - **`normalized`** `instance of pandas.core.DataFrame`

The normalized data.

#### `plot3DMatrix(data, picks, trialList, average, fs)`

durSamples = int(np.round(dur*fs))
baselineDur = 0.1
baselineDurSamples = int(np.round(baselineDur*fs))
# subset trials
dataSub = data[trialList,:,:]

# subset electrodes
dataSub = dataSub[:,:,picks]

# calculate mean across trials
print 'Averaging %d trials...' % (dataSub.shape[0])
dataMean = np.mean(dataSub, axis=0)

#     # plot time sequence
#     x = np.arange(-baselineDurSamples, durSamples)/fs
#     plt.figure()
#     plt.plot(x, dataMean[:durSamples+baselineDurSamples,:])
#     plt.axvline(x=0)
#     plt.show()
#     plt.close()

# plot fft
plotFFTNP(dataMean, average=average, fs=fs)

plotDataSubset(data, stim, events, offset, t0=0, t1=1, fs=2048.):

#### `plotDataSubset(data, stim, events, offset, t0=0, t1=1, fs=2048.)`

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

Returns
-------
   - **`fig`** `instance of matplotlib.figure.Figure`

The figure of the data subset in the time domain.

#### `plotERPElectrodes(data, trialNumList, events, trialDur=None, fs=2048.`

startOffset=0):

#### `plotFFT(data, facet=False, freqMin=None, freqMax=None, yMin=None`


#### `plotFFTElectrodes(data, trialNumList, events, trialDur, fs`

freqMin=None, freqMax=None, yMin=None, yMax=None, startOffset=0, noiseAve=None):

#### `plotFFTNP(data, average, fs)`

data = data.mean(axis=1)

fAx, fftData = computeFFT(data, fs)

plt.figure()
plt.plot(fAx, fftData, linewidth=0.5)
plt.xlabel('frequency (Hz)')
plt.xticks([4, 7, 13, 26])
plt.xlim(0, 35)
plt.show()

plotFilterResponse(zpk, fs):

#### `plotFilterResponse(zpk, fs)`

Plot the filter frequency response.

- **`zpk`** `array-like`

   The 3 parameters of the filter [z, p, k].
- **`fs`** `float`

   Sampling frequency in Hz.

Returns
-------
   - **`fig`** `instance of matplotlib.figure.Figure`

The figure of the filter response.

#### `preprocessing(files)`

data = createDataFromRaw(raw)
print 'keeping eeg...'
# Keep only eeg channels
eegData = data.iloc[:, 1:65]
print 'creating stim'
# add stim channel (named 'STI 014' with biosemi)
stim = data['STI 014']
print 'getting events...'
# Get events
startEvents = getEvents(raw=raw, eventCode=65284)
startSound = getEvents(raw=raw, eventCode=65288)
# Use discriminate events because two types are on the same channel
startSound = discriminateEvents(startSound, 300)

return raw, data, eegData, stim, startEvents, startSounds

getBehaviorTables(dbAddress, dbName):
tableHJ1 = getBehaviorData(dbAddress, dbName, sessionNum=141)

# tableHJ2 is in split in two sessions
tableHJ2 = getBehaviorData(dbAddress, dbName, sessionNum=144)
tableHJ2_secondSession = getBehaviorData(dbAddress, dbName, sessionNum=145)
tableHJ2_secondSession['trialNum'] += 81
tableHJ2 = tableHJ2.append(tableHJ2_secondSession)

tableHJ3 = getBehaviorData(dbAddress, dbName, sessionNum=147)
return tableHJ1, tableHJ2, tableHJ3

mergeBehaviorTables(tableHJ1, tableHJ2, tableHJ3):
tableHJ1Temp = tableHJ1.copy()
tableHJ1Temp['session'] = 1
# Last trial of first session: trigger but no answer so no behavior response
# Add a fake behavioral trial
lenHJ1 = tableHJ1Temp.shape[0]
tableHJ1Temp.loc[lenHJ1+1] = [lenHJ1, 0, 0, 0, 1, 0]

tableHJ2Temp = tableHJ2.copy()
tableHJ2Temp['trialNum'] += lenHJ1 + 1
tableHJ2Temp['session'] = 2

tableHJ3Temp = tableHJ3.copy()
tableHJ3Temp['trialNum'] += lenHJ1 + 1 + tableHJ2.shape[0] 
tableHJ3Temp['session'] = 3

tableAll = tableHJ1Temp.append([tableHJ2Temp, tableHJ3Temp], ignore_index=True)
tableAll = tableAll.sort_values('trialNum')
tableAll = tableAll.reset_index(drop=True)

return tableAll


#### `refToAverageNP(data)`

print average
average = average.reshape(average.shape[0], 1)
newData = data - average
return newData

refToMastoids(data, M1, M2):

#### `refToMastoids(data, M1, M2)`

Transform each electrode of data according to the average of M1 and M2.

- **`data`** `instance of pandas.core.DataFrame`

   First column has to contain the timing of events in frames.
- **`M1`** `instance of pandas.core.series.Series`

   Values of mastoid 1. This Series has to be the same length as data.
- **`M2`** `instance of pandas.core.series.Series`

   Values of mastoid 2. This Series has to be the same length as data

Returns
-------
   - **`newData`** `instance of pandas.core.DataFrame`

A dataframe referenced to matoids containing all electrode from which
we subtract the average of M1 and M2.

#### `refToMastoidsNP(data, M1, M2)`

mastoidsMean = mastoidsMean.reshape(mastoidsMean.shape[0], 1)
print mastoidsMean.shape
newData = data - mastoidsMean
return newData


########## FUNCTIONS FOR THE PILOT SESSION ###################
compareTimeBehaviorEEG(dbAddress, dbName, events, startSound, interTrialDur,
sessionNum=None, table=None, fs=2048.):