import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pylab import *
from matplotlib import gridspec

import mne
from mne import create_info, find_events, Epochs, pick_types
from mne.time_frequency import tfr_multitaper

from scipy import signal, fftpack
from scipy.signal import butter, cheby2, filtfilt, resample, lfilter, freqz, freqz_zpk, zpk2sos, sosfilt, sosfiltfilt, cheb2ord

import couchdb

import h5py

def addOffset(data, offset):
    """
    Plot all electrodes with an offset from t0 to t1. The stimulus channel is
    also ploted and red lines are used to show the events.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
       Add offset to data.
    offset : float
        Value of the offset.

    Returns
    -------
    newData : instance of pandas.core.DataFrame
        The data with offset applied to each electrode.
    """

    newData = data + offset * np.arange(data.shape[1]-1,-1,-1)
    return newData

def calculateBaseline(data, baselineDur=0.1, fs=2048.):
    """
    Calculate and return the baseline (average of each data point) of a signal.
    The baseline will calculated from the first `baselineDur` seconds of this
    signal.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data used to calculate the baseline.
    baselineDur : float
        Duration of the baseline to use for the calulation of the average in
        seconds.
    fs : float
        Sampling frequency of data in Hz.

    Returns
    -------
    baseline : float
        The baseline value.
    """

    # duration for the baseline calculation is in seconds
    durSamples = int(np.round(baselineDur*fs))
    subData = data[:durSamples]
    baseline = subData.mean()
    return baseline

def chebyBandpassFilter(data, cutoff, gstop=40, gpass=1, fs=2048.):
    """
    Design a filter with scipy functions avoiding unstable results (when using
    ab output and filtfilt(), lfilter()...).
    Cf. ()[]

    Parameters
    ----------
    data : instance of numpy.array | instance of pandas.core.DataFrame
        Data to be filtered. Each column will be filtered if data is a
        dataframe.
    cutoff : array-like of float
        Pass and stop frequencies in order:
            - the first element is the stop limit in the lower bound
            - the second element is the lower bound of the pass-band
            - the third element is the upper bound of the pass-band
            - the fourth element is the stop limit in the upper bound
        For instance, [0.9, 1, 45, 48] will create a band-pass filter between
        1 Hz and 45 Hz.
    gstop : int
        The minimum attenuation in the stopband (dB).
    gpass : int
        The maximum loss in the passband (dB).

    Returns
    -------
    filteredData : instance of numpy.array | instance of pandas.core.DataFrame
        The filtered data.
    """

    wp = [cutoff[1]/(fs/2), cutoff[2]/(fs/2)]
    ws = [cutoff[0]/(fs/2), cutoff[3]/(fs/2)]

    z, p, k = signal.iirdesign(wp = wp, ws= ws, gstop=gstop, gpass=gpass,
        ftype='cheby2', output='zpk')
    zpk = [z, p, k]
    sos = zpk2sos(z, p, k)

    order, Wn = cheb2ord(wp = wp, ws= ws, gstop=gstop, gpass=gpass, analog=False)
    print 'Creating cheby filter of order %d...' % order

    if (data.ndim == 2):
        print 'Data contain multiple columns. Apply filter on each columns.'
        filteredData = np.zeros(data.shape)
        for electrode in range(data.shape[1]):
            # print 'Filtering electrode %s...' % electrode
            filteredData[:, electrode] = sosfiltfilt(sos, data[:, electrode])
    else:
        # Use sosfiltfilt instead of filtfilt fixed the artifacts at the beggining
        # of the signal
        filteredData = sosfiltfilt(sos, data)
    return zpk, filteredData

def checkPlots(data1, data2, fs1, fs2, start, end, electrodeNum):
    """
    Check filtering and downsampling by ploting both datasets.

    Parameters
    ----------
    data1 : instance of pandas.core.DataFrame
        First dataframe.
    data2 : instance of pandas.core.DataFrame
        Second dataframe.
    fs1 : float
        Sampling frequency of the first dataframe in Hz.
    fs2 : float
        Sampling frequency of the second dataframe in Hz.
    start : float
        Start of data to plot in seconds.
    end : float
        End of data to plot in seconds.
    electrodeNum : int
        Index of the column to plot.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure containing both dataset plots.
    """

    start1 = int(np.round(fs1*start))
    start2 = int(np.round(fs2*start))
    end1 = int(np.round(fs1*end))
    end2 = int(np.round(fs2*end))
    # Choose electrode and time to plot
    data1Sub = data1.iloc[start1: end1, electrodeNum]
    data2Sub = data2.iloc[start2: end2, electrodeNum]
    # x-axis in seconds
    x1 = np.arange(data1Sub.shape[0])/fs1
    x2 = np.arange(data2Sub.shape[0])/fs2

    plt.figure()
    plt.plot(x1, data1Sub)
    plt.plot(x2, data2Sub, alpha=0.7)
    plt.show()

def checkPlotsNP(data1, data2, fs1, fs2, start, end, electrodeNum):
    """
    Check filtering and downsampling by ploting both datasets.

    Parameters
    ----------
    data1 : instance of pandas.core.DataFrame
        First dataframe.
    data2 : instance of pandas.core.DataFrame
        Second dataframe.
    fs1 : float
        Sampling frequency of the first dataframe in Hz.
    fs2 : float
        Sampling frequency of the second dataframe in Hz.
    start : float
        Start of data to plot in seconds.
    end : float
        End of data to plot in seconds.
    electrodeNum : int
        Index of the column to plot.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure containing both dataset plots.
    """

    start1 = int(np.round(fs1*start))
    start2 = int(np.round(fs2*start))
    end1 = int(np.round(fs1*end))
    end2 = int(np.round(fs2*end))
    # Choose electrode and time to plot
    data1Sub = data1[start1:end1, electrodeNum]
    data2Sub = data2[start2:end2, electrodeNum]
    # x-axis in seconds
    x1 = np.arange(data1Sub.shape[0])/fs1
    x2 = np.arange(data2Sub.shape[0])/fs2

    plt.figure()
    plt.plot(x1, data1Sub)
    plt.plot(x2, data2Sub, alpha=0.7)
    plt.show()

def computeFFT(data, fs):
    """
    Compute the FFT of `data` and return also the axis in Hz for further plot.

    Parameters
    ----------
    data : array
        First dataframe.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    fAx : instance of numpy.array
        Axis in Hz to plot the FFT.
    fftData : instance of numpy.array
        Value of the fft.
    """
    N = data.shape[0]
    fAx = np.arange(N/2)*fs/N
    Y = np.abs(fftpack.fft(data, axis=0))
    fftData = 2.0/N * np.abs(Y[0:N//2])
    return fAx, fftData

def computePickEnergy(data, pickFreq, showPlot, fs):
    """
    Calculate the relative energy at the frequency `pickFreq` from the the FFT
    of `data`. Compare the mean around the pick with the mean of a broader zone
    for each column.

    Parameters
    ----------
    data : array-like
        Matrix of the shape (time, electrode).
    pickFreq : float
        Frequency in Hz of the pick for which we want to calculate the relative energy.
    showPlot : boolean
        A plot of the FFT can be shown.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    pickRatio : float
        Relative energy of the pick.
    """
    N = data.shape[0]
    fAx, fftData = computeFFT(data, fs)

    # Convert pick from Hz to bin number
    pickBin = int(np.round(pickFreq*(N/fs)))

    pickData = fftData[pickBin:pickBin+1]
    pickDataMean = pickData.mean(axis=0)

    nonPickBin = np.concatenate([np.arange(pickBin-5, pickBin),
                                np.arange(pickBin+1, pickBin+5+1)]);

    nonPickData = fftData[nonPickBin]
    nonPickDataMean = nonPickData.mean(axis=0)

    pickRatio = pickDataMean / nonPickDataMean

    if (showPlot):
        plt.figure()
        plt.plot(fAx, fftData, linewidth=0.5)
        plt.show()

    # meanColumn = data.mean(axis=0)
    return pickRatio

def create3DMatrix(data, trialTable, events, trialList, fs):
    trials = trialTable.copy()
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

def createStimChannel(events):
    """
    Create stim channel from events.

    Parameters
    ----------
    events : instance of pandas.core.DataFrame
        Dataframe containing list of events obtained with mne.find_events(raw)
       .

    Returns
    -------
    stim : instance of pandas.core.series.Series
        Series containing the stimulus channel reconstructed from events.
    """

    newEvents = events.copy()
    newEvents[0] = newEvents[0].round(0).astype('int')
    chan = np.zeros(int(events.iloc[-1, 0]))
    chan[newEvents.iloc[:-2, 0]] = 8
    stim = pd.Series(chan, columns = ['STI 014'])
    return stim

def discriminateEvents(events, threshold):
    """
    Discriminate triggers when different kind of events are on the same channel.
    A time threshold is used to determine if two events are from the same trial.

    Parameters
    ----------
    events : instance of pandas.core.DataFrame
        Dataframe containing the list of events obtained with
        mne.find_events(raw).
    threshold : float
        Time threshold in milliseconds. Keeps an event if the time difference
        with the next one is superior than threshold.

    Returns
    -------
    newData : instance of pandas.series.Series
        List of trial number filling the requirements.
    """

    # calculate the rolling difference (between n and n+1)
    events['diff'] = events[0].diff()
    # replace the nan with the first value
    events['diff'].iloc[0] = events.iloc[0, 0]
    # select events with time distance superior to threshold
    events = events[events['diff']>threshold]
    events = events.reset_index(drop=True)
    del events['diff']
    return events

def downsample(data, oldFS, newFS):
    """
    Resample data from oldFS to newFS using the scipy 'resample' function.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data to resample.
    oldFS : float
        The sampling frequency of data.
    newFS : float
        The new sampling frequency.

    Returns
    -------
    newData : instance of pandas.DataFrame
        The downsampled dataset.
    """

    newNumSamples = int((data.shape[0] / oldFS) * newFS)
    newData = pd.DataFrame(resample(data, newNumSamples))
    return newData

def downsampleEvents(events, oldFS, newFS):
    """
    Modify the timestamps of events to match a new sampling frequency.

    Parameters
    ----------
    events : instance of pandas.core.DataFrame
        Dataframe containing list of events obtained with mne.find_events(raw)
       .
    oldFS : float
        The sampling frequency of the input events.
    newFS : float
        The sampling frequency to the output events.

    Returns
    -------
    newEvents : instance of pandas.DataFrame
        DataFrame containing the downsampled events.
    """

    newEvents = events.copy()
    newEvents[0] = (events[0]/oldFS)*newFS
    newEvents[0] = newEvents[0].round(0).astype('int')
    return newEvents

def downsampleNP(data, oldFS, newFS):
    """
    Resample data from oldFS to newFS using the scipy 'resample' function.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data to resample.
    oldFS : float
        The sampling frequency of data.
    newFS : float
        The new sampling frequency.

    Returns
    -------
    newData : instance of pandas.DataFrame
        The downsampled dataset.
    """

    newNumSamples = int((data.shape[0] / oldFS) * newFS)
    newData = resample(data, newNumSamples)
    return newData

def FFTTrials(data, events, trialNumList, baselineDur, trialDur, fs, normalize,
    electrodes):
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

def getBehaviorData(dbAddress, dbName, sessionNum):
    """
    Fetch behavior data from couchdb (SOA, SNR and trial duration).

    Parameters
    ----------
    dbAddress : str
        Path to the couch database.
    dbName : str
        Name of the database on the couch instance.
    sessionNum : int
        Behavior data will be fetched from this sessionNum.

    Returns
    -------
    lookupTable : instance of pandas.core.DataFrame
        A dataframe containing trial data.
    """

    couch = couchdb.Server(dbAddress)
    db = couch[dbName]
    lookupTable = pd.DataFrame(columns=['trialNum', 'SOA', 'SNR', 'targetRate',
        'targetFreq','trialDur', 'soundStart', 'deviant', 'noise'])
    count = 0
    for docid in db.view('_all_docs'):
        if (docid['id'].startswith('infMask_%d' % sessionNum)):
            count += 1

            trialNum = int(docid['id'].split('_')[-1])

            if (db.get(docid['id'])['toneCloudParam'] is not None):
                doc = pd.DataFrame(db.get(docid['id']))

                toneCloudLen = doc.toneCloudParam.shape[0]
                trialDur = doc.trialDur[0]
                toneCloud = pd.DataFrame(db.get(docid['id'])['toneCloudParam'])
                soundStart = np.min(toneCloud['time'])
                deviant = doc.deviant[0]
                recoverNoise = doc.toneCloudParam[300]['gain']!=0
                if ('targetSOA' in doc.columns):
                    targetRate = 1/doc.targetSOA[0]
                else:
                    targetRate = None
                targetFreq = doc.targetFreq[0]
                SNR = doc.targetdB[0]
                # target SOA can be infered from the number of tone in the cloud
                if (toneCloudLen < 700):
                    recoverTargetSOA = 4
                elif ((toneCloudLen > 800) & (toneCloudLen < 1100)):
                    recoverTargetSOA = 7
                elif (toneCloudLen > 1300):
                    recoverTargetSOA = 13
                else:
                    raise ValueError('check the value of toneCloudLen')

            else:
                doc = pd.Series(db.get(docid['id']))
                trialDur = doc.trialDur
                deviant = doc.deviant
                targetFreq = doc.targetFreq
                SNR = doc.targetdB
                targetRate = 1/doc.targetSOA
                soundStart = 0
                recoverTargetSOA = None
                recoverNoise = doc.noise


            lookupTable.loc[count] = pd.Series({'trialNum': trialNum,
                    'SOA': recoverTargetSOA, 'SNR': SNR, 'targetRate': targetRate,
                    'targetFreq': targetFreq, 'trialDur': trialDur,
                    'soundStart': soundStart, 'deviant': deviant,
                    'noise': recoverNoise})

    return lookupTable

def getEvents(raw, eventCode):
    """
    Get the events corresponding to `eventCode`.

    Parameters
    ----------
    raw : instance of mne.io.edf.edf.RawEDF
        RawEDF object from the MNE library containing data from the .bdf files.
    eventCode : int
        Code corresponding to a specific events. For instance, with a biosemi
        device, the triggers are coded 65284, 65288 and 65296 respectively on
        the first, second and third channel.

    Returns
    -------
    startEvents : instance of pandas.core.DataFrame
        Dataframe containing the list of timing corresponding to the event code
        in the first column. The second column contains the code before the event
        and the third the code of the selected event.
    """

    events = mne.find_events(raw)
    eventsDf = pd.DataFrame(events)

    # Keep only event corresponding to the event code
    startEvents = eventsDf.loc[eventsDf[2]==eventCode]

    startEvents = startEvents.set_index([np.arange(len(startEvents))])
    startEvents.index.name = 'start'
    return startEvents

def getTrialsAverage(data, events, trialDur=None, trialNumList=None,
    baselineDur=0.1, normalize=False, fs=2048., startOffset=0, noiseAve=None):
    """
    Get the average across trials (from `trialNumList`) based on time-locking
    provided by `events`.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data containing values across time (not epoched).
    events : instance of pandas.core.DataFrame
        Dataframe containing the list of events obtained with
        mne.find_events(raw).
    trialDur : float | None
        Trial duration in seconds.
    trialNumList : array-like of int | None
        List of all trials to use. If None, all trials are taken.
    baselineDur : float, defaults to 0.1
        Duration of the baseline in seconds. If normalize is True, normalize
        each electrode with a baseline of duration `baselineDur`.
    normalize : bool, defaults to False
        If True data will be normalized.
    fs : float
        Sampling frequency of data in Hz.

    Returns
    -------
    meanTrials : instance of pandas.series.Series
        Series containing the averaged values across trials.
    allTrials : instance of pandas.core.DataFrame
        Dataframe containing the values of each trial (1 column = 1 trial).
    """

    # Calculate average across trials for this Series
    if (trialNumList is None):
        trialNumList = [events.shape[0]]

    allTrials = pd.DataFrame()
    for trialNum in trialNumList:
        trialData = getTrialData(data, events=events, trialNum=trialNum,
            baselineDur=baselineDur, trialDur=trialDur, fs=fs,
            startOffset=startOffset)
        if normalize:
            trialData = normalizeFromBaseline(trialData,
                baselineDur=baselineDur, fs=fs)

        trialData = trialData.reset_index(drop=True)

        if noiseAve is not None:
            trialData = trialData - noiseAve

        allTrials['trial%d' % trialNum] = pd.Series(trialData)

    # convert baselineDur in frames
    start = int(np.round(baselineDur *fs))

    # Change index to have x-axis in seconds
    allTrials = allTrials.set_index(np.arange(-start, allTrials.shape[0]-start)/fs)

    meanTrials = allTrials.mean(axis=1)
    return meanTrials, allTrials

def getTrialData(data, events, trialNum=0, electrode=None, baselineDur=0.1,
    trialDur=None, fs=2048., startOffset=0):
    """
    Get the epochs from data (time series containing all epochs/trials) for the
    trial `trialNum`.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data containing values across time (not epoched).
    events : instance of pandas.core.DataFrame
        Dataframe containing the list of events obtained with
        mne.find_events(raw).
    trialNum : int, defaults to 0
        Trial number. The returned epoch corresponds to this trial number.
    electrode : str, default to None
        The epoch will returned only for this electrode. If None, all electrodes
        will be used.
    baselineDur : float, defaults to 0.1
        Duration of the baseline in seconds. The returned epoch contains this
        duration at the beginning for further use (normalization, plots...).
    trialDur : float | None
        Trial duration in seconds. If None, the whole trial duration will be
        used.
    fs : float
        Sampling frequency of data in Hz.

    Returns
    -------
    dataElectrode : instance of pandas.core.DataFrame
        Dataframe containing 1 trial from `data` for every or 1 electrode.

    """
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
        dataElectrode = data.iloc[start:end]
    else:
        dataElectrode = data.iloc[start:end][electrode]

    return dataElectrode

def getTrialDataNP(data, events, trialNum=0, electrode=None, baselineDur=0.1,
    trialDur=None, fs=2048., startOffset=0):
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

def getTrialNumList(table, **kwargs):
    """
    Returns a subset of table according to SOA, SNR and/or targetFreq. This is
    used to select trials with specific parameters.

    Parameters
    ----------
    table : instance of pandas.core.DataFrame
        DataFrame containing trial number and their parameters (SOA, SNR...).
    kwargs : array-like of int | None
        Array containing element from table to select. It can be `SOA`, `SNR` or
        `targetFreq`.

    Returns
    -------
    newData : instance of pandas.series.Series
        List of trial number filling the requirements.
    """

    if (kwargs):
        acc = pd.DataFrame()
        for i in kwargs:
            acc[i] = table[i].isin(kwargs[i])
        acc['res'] = acc.apply(mean, axis=1)
        print kwargs
        return table.trialNum[acc['res']==1]
    else:
        print 'All SOA and all SNR'
        return table.trialNum

def importH5(name, df):
    f = h5py.File(name,'r') 
    data = f.get(df)

    data = np.array(data)
    oldShape = data.shape
    data = np.swapaxes(data, 1, 2)
    print 'convert shape %s to %s' % (oldShape, data.shape)
    return data

def loadEEG(path):
    """
    Load data from .bdf files. If an array of path is provided, files will be
    concatenated.

    Parameters
    ----------
    path : str | array-like of str
        Path to the .bdf file(s) to load.

    Returns
    -------
    raw : instance of mne.io.edf.edf.RawEDF
        RawEDF object from the MNE library containing data from the .bdf files.
    """

    if isinstance(path, list):
        temp = []
        for i in path:
            data = mne.io.read_raw_edf(i)
            temp.append(data)
        print temp
        raw = mne.concatenate_raws(temp)
    else:
        raw = mne.io.read_raw_edf(path)
    return raw

def normalizeFromBaseline(data, baselineDur=0.1, fs=2048.):
    """
    Normalize data by subtracting the baseline to each data point. The data used
    to normalize has to be included at the beginning of data. For instance, to
    normalize a 10 seconds signal with a 0.1 second baseline, data has to be
    10.1 seconds and the baseline used will be the first 0.1 second.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data to normalize.
    baselineDur : float
        Duration of the baseline to use for the normalization in seconds.
    fs : float
        Sampling frequency of data in Hz.

    Returns
    -------
    normalized : instance of pandas.core.DataFrame
        The normalized data.
    """

    start = int(np.round(baselineDur*fs))
    baseline = calculateBaseline(data, baselineDur=baselineDur, fs=fs)
    normalized = data-baseline
    return normalized

def plot3DMatrix(data, picks, trialList, average, fs):
    dur = 10
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

def plotDataSubset(data, stim, events, offset, t0=0, t1=1, fs=2048.):
    """
    Plot all electrodes with an offset from t0 to t1. The stimulus channel is
    also ploted and red lines are used to show the events.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data to plot (not epoched). Columns correspond to electrodes.
    stim : instance of pandas.core.DataFrame
        One column dataframe containing the event codes. Used to plot the
        stimulus timing along with EEG.
    events : instance of pandas.core.DataFrame
        Dataframe containing the list of events obtained with
        mne.find_events(raw).
    offset : float
        Offset between each electrode line on the plot.
    t0 : float
        Start of data to plot.
    t1 : float
        End of data to plot.
    fs : float
        Sampling frequency of data in Hz.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure of the data subset in the time domain.
    """

    # Subset
    start = int(np.round(fs*t0))
    end = int(np.round(fs*t1))
    subData = data[start:end]
    # spread lines equally on the y-axis
    subData = addOffset(subData, offset=offset)
    stim = stim.iloc[start:end]
    # scale the stim channel to see it on the plot
    stim = (stim-65280.0)
    # x-axis in seconds
    subData = subData.set_index(subData.index/fs)

    # add time for stim
    stim = pd.DataFrame(stim)
    stim = stim.set_index(subData.index.values)

    plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1])

    plt.subplot(gs[0])
    plt.plot(subData)
    for i in events[0]:
        if (i>start and i<end):
            print 'Sound starts at %f seconds (red vertical line)' % (i/fs)
            plt.axvline(i/fs)
    plt.legend(subData.columns, bbox_to_anchor=(1, 1), ncol=4)

    plt.subplot(gs[1])
    plt.plot(stim)
    plt.legend(stim.columns, bbox_to_anchor=(1, 1), ncol=4)
    plt.show()
    plt.close()

def plotERPElectrodes(data, trialNumList, events, trialDur=None, fs=2048.,
    baselineDur=0.1, electrodes='Fp1', normalize=False, facet=False,
    startOffset=0):
    """
    Plot the ERP (average across trials time-locked to specific events) of
    each electrode as single lines on the same figure with or without facetting.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data containing the time series to transform and plot. Each column is an
        electrode.
    trialNumList : array-like of int
        List of all trials to use to compute the FFT.
    events : instance of pandas.core.DataFrame
        Dataframe containing the list of events obtained with
        mne.find_events(raw).
    trialDur : float
        Trial duration in seconds.
    fs : float
        Sampling frequency of data in Hz.
    baselineDur : float, defaults to 0.1
        Duration of the baseline in seconds. If normalize is True, normalize
        each electrode with a baseline of duration `baselineDur`.
    electrodes : int | array-like of int, default to 'Fp1'
        List of electrodes to use to compute the FFT.
    normalize : bool, defaults to False
        If True data will be normalized.
    facet : bool, default to False
        If True, each electrode will be plotted on a different facet.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure of the ERP.
    """

    print 'Average of %d trials' % len(trialNumList)
    meanTrials = pd.DataFrame()
    for electrode in electrodes:
        meanTrials[electrode], allTrials = getTrialsAverage(data=data[electrode],
            events=events, trialDur=trialDur, trialNumList=trialNumList,
            baselineDur=baselineDur, normalize=normalize, fs=fs, startOffset=startOffset)

    if (facet):
        print 'Faceting...'
        meanTrials.plot(subplots=True)
    else:
        plt.figure()
        plt.plot(meanTrials)
        plt.axvline(x=0, color='grey', linestyle='dotted')
        plt.axvspan(-baselineDur, 0, alpha=0.3, color='grey')
        plt.xlabel('Time (s)')
        # plt.legend(meanTrials.columns, bbox_to_anchor=(1, 1), ncol=4)
        plt.show()

def plotFFT(data, facet=False, freqMin=None, freqMax=None, yMin=None,
    yMax=None, fs=2048.):
    """
    Create the x-axis and plot the FFT of data.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame, shape()
        Data containing the frequency series to plot. Each column is an
        electrode.
    facet : bool, default to False
        If True, each electrode will be plotted on a different facet.
    freqMin : float, default to None
        Minimum frequency (x-axis) to show on the plot.
    freqMax : float, default to None
        Maximum frequency (x-axis) to show on the plot.
    yMin : float, default to None
        Minimum value (y-axis) to show on the plot.
    yMax : float, default to None
        Maximum value (y-axis) to show on the plot.
    fs : float
        Sampling frequency of data in Hz.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure of the FFT.
    """

    N = data.shape[0]
    Ts = 1.0/fs; # sampling interval
    t = np.arange(0, N/fs, Ts) # time vector
    tf = np.linspace(0.0, 1.0/(2.0*Ts), N//2)
    newMeanTrials = 2.0/N * np.abs(data.iloc[0:N//2, :])
    newMeanTrials = newMeanTrials.set_index(tf)
    if (facet):
        newMeanTrials.plot(ylim=(yMin, yMax), xlim=(freqMin, freqMax),subplots=True)
        plt.xlabel('frequency (Hz)')
        plt.xticks([4, 7, 13, 26])
    else:
        plt.figure()
        plt.plot(newMeanTrials, linewidth=0.5)
        if (freqMin is not None):
            plt.xlim(left=freqMin)
        if (freqMax is not None):
            plt.xlim(right=freqMax)
        if (yMin is not None):
            plt.ylim(bottom=yMin)
        if (yMax is not None):
            plt.ylim(top=yMax)
        # plt.legend(data.columns, bbox_to_anchor=(1, 1), ncol=4)
        plt.xlabel('frequency (Hz)')
        plt.xticks([4, 7, 13, 26])
        # plt.show()

def plotFFTElectrodes(data, trialNumList, events, trialDur, fs,
    baselineDur=0.1, electrodes='Fp1', normalize=False, facet=False,
    freqMin=None, freqMax=None, yMin=None, yMax=None, startOffset=0, noiseAve=None):
    """
    Plot the FFT of each electrode as single lines on the same figure with or
    without facetting. The FFT is computed from the ERP (average across trials
    time-locked to specific events).

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data containing the time series to transform and plot. Each column is an
        electrode.
    trialNumList : array-like of int
        List of all trials to use to compute the FFT.
    events : instance of pandas.core.DataFrame
        Dataframe containing the list of events obtained with
        mne.find_events(raw).
    trialDur : float
        Trial duration in seconds.
    fs : float
        Sampling frequency of data in Hz.
    baselineDur : float, defaults to 0.1
        Duration of the baseline in seconds. If normalize is True, normalize
        each electrode with a baseline of duration `baselineDur`.
    electrodes : int | array-like of int, default to 'Fp1'
        List of electrodes to use to compute the FFT.
    normalize : bool, defaults to False
        If True data will be normalized.
    facet : bool, default to False
        If True, each electrode will be plotted on a different facet.
    freqMin : float, default to None
        Minimum frequency (x-axis) to show on the plot.
    freqMax : float, default to None
        Maximum frequency (x-axis) to show on the plot.
    yMin : float, default to None
        Minimum value (y-axis) to show on the plot.
    yMax : float, default to None
        Maximum value (y-axis) to show on the plot.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure of the FFT.
    """

    print 'Average of %d trials' % len(trialNumList)
    allTrials = pd.DataFrame()
    baselineDurSamples = int(np.round(baselineDur))
    for electrode in electrodes:
        meanDataElectrode, allTrialsElectrode = getTrialsAverage(data=data[electrode], events=events,
            trialDur=trialDur, trialNumList=trialNumList, baselineDur=baselineDur,
            normalize=normalize, fs=fs, startOffset=startOffset)
        Y = fftpack.fft(meanDataElectrode[baselineDurSamples:])
        allTrials[electrode] = pd.Series(np.abs(Y))

    plotFFT(allTrials, facet=facet, freqMin=freqMin, freqMax=freqMax,
        yMin=yMin, yMax=yMax, fs=fs)

def plotFFTNP(data, average, fs):
    if average:
        data = data.mean(axis=1)

    fAx, fftData = computeFFT(data, fs)

    plt.figure()
    plt.plot(fAx, fftData, linewidth=0.5)
    plt.xlabel('frequency (Hz)')
    plt.xticks([4, 7, 13, 26])
    plt.xlim(0, 35)
    plt.show()

def plotFilterResponse(zpk, fs):
    """
    Plot the filter frequency response.

    Parameters
    ----------
    zpk : array-like
        The 3 parameters of the filter [z, p, k].
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure of the filter response.
    """

    z, p, k = zpk
    w, h = freqz_zpk(z, p, k, worN=8000)

    plt.plot(0.5*fs*w/np.pi, 20 * np.log10(abs(h)), 'b')
    plt.title('Chebyshev II bandstop filter')
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude [dB]')

def refToAverageNP(data):
    average = np.mean(data, axis=1)
    print average
    average = average.reshape(average.shape[0], 1)
    newData = data - average
    return newData

def refToMastoids(data, M1, M2):
    """
    Transform each electrode of data according to the average of M1 and M2.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        First column has to contain the timing of events in frames.
    M1 : instance of pandas.core.series.Series
        Values of mastoid 1. This Series has to be the same length as data.
    M2 : instance of pandas.core.series.Series
        Values of mastoid 2. This Series has to be the same length as data

    Returns
    -------
    newData : instance of pandas.core.DataFrame
        A dataframe referenced to matoids containing all electrode from which
        we subtract the average of M1 and M2.
    """

    mastoids = pd.concat([M1, M2], axis=1)
    mastoidsMean = mastoids.mean(axis='columns')
    print mastoidsMean.shape
    newData = data.sub(mastoidsMean, axis=0)
    return newData

def refToMastoidsNP(data, M1, M2):
    mastoidsMean = np.mean([M1, M2], axis=0)
    mastoidsMean = mastoidsMean.reshape(mastoidsMean.shape[0], 1)
    print mastoidsMean.shape
    newData = data - mastoidsMean
    return newData


############## FUNCTIONS FOR THE PILOT SESSION ###################
def compareTimeBehaviorEEG(dbAddress, dbName, events, startSound, interTrialDur,
    sessionNum=None, table=None, fs=2048.):
    """
    Compare trial's durations from behavioral data (stored in a couchDB) and
    from the list of events (extracted from the stim channel). This can be used
    to assess the triggers accuracy.

    Parameters
    ----------
    dbAddress : str
        Path to the couch database.
    dbName : str
        Name of the database on the couch instance.
    events : instance of pandas.core.DataFrame
        Dataframe containing the list of events (trial start time) obtained with
        mne.find_events(raw).
    startSound : instance of pandas.core.DataFrame
        Dataframe containing the list of events (sound start time) obtained with
        mne.find_events(raw).
    sessionNum : int | None
        If None table argument has to be provided and will be used as behavior
        data. Otherwise behavior data will be fetched from this sessionNum.
    table: pandas DataFrame | None
        if None sessionNum will be used to fetch behavior data otherwise
        behavior data is given directly in table.
    fs: float
        Sampling frequency of `startSound` and `startEvents`.

    Returns
    -------
    comparisonTable : instance of pandas.core.DataFrame
        A dataframe containing columns ['trialNum', 'trialDur_eeg (ms)',
        'trialDur_behavior (ms)', 'diff (ms)', 'soundStart (ms)',
        'soundStart_eeg (ms)', 'diffSound (ms)'].
    """
    if (table is not None):
        lookupTable = table
    else:
        lookupTable = getBehaviorData(dbAddress, dbName, sessionNum)
    comparisonTable = pd.DataFrame(columns=['trialNum', 'trialDur_eeg',
        'trialDur_behavior', 'diff', 'soundStart', 'soundStart_eeg', 'diffSound'])
    eventsCopy = events.reset_index()
    # go from samples to ms
    eventsCopy[0] = eventsCopy[0]*1000/fs
    startSoundCopy = startSound.reset_index()
    startSoundCopy[0] = startSoundCopy[0]*1000/fs
    count = 0
    for i in range(eventsCopy.shape[0]-1):
        # comparisonTable['trialNum'] = lookupTable['trialNum']
        trialDur_eeg = (eventsCopy[0][i+1] - eventsCopy[0][i]) - interTrialDur
        trialDur_behavior = lookupTable['trialDur'][lookupTable['trialNum']==i].iloc[0]

        soundStart = lookupTable['soundStart'][lookupTable['trialNum']==i].iloc[0]*1000
        # If starting tone cloud is more than 100 ms, the target was the first
        # sound heard by the participant and it is played 100 ms after the start
        if (soundStart>100):
            soundStart = 100

        soundStart_eeg = startSoundCopy[0][i] - eventsCopy[0][i]

        diffSound = np.abs(soundStart_eeg - soundStart)

        diff = np.abs(trialDur_eeg - trialDur_behavior)
        comparisonTable.loc[count] = pd.Series({'trialNum': i,
            'trialDur_eeg': trialDur_eeg,
            'trialDur_behavior': trialDur_behavior, 'diff': diff,
            'soundStart': soundStart, 'soundStart_eeg': soundStart_eeg,
            'diffSound': diffSound})
        count += 1

    comparisonTable.columns = ['trialNum', 'trialDur_eeg (ms)',
        'trialDur_behavior (ms)', 'diff (ms)', 'soundStart (ms)',
        'soundStart_eeg (ms)', 'diffSound (ms)']
    return comparisonTable

def preprocessing(files):
    raw = loadEEG(files)
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

def getBehaviorTables(dbAddress, dbName):
    tableHJ1 = getBehaviorData(dbAddress, dbName, sessionNum=141)

    # tableHJ2 is in split in two sessions
    tableHJ2 = getBehaviorData(dbAddress, dbName, sessionNum=144)
    tableHJ2_secondSession = getBehaviorData(dbAddress, dbName, sessionNum=145)
    tableHJ2_secondSession['trialNum'] += 81
    tableHJ2 = tableHJ2.append(tableHJ2_secondSession)

    tableHJ3 = getBehaviorData(dbAddress, dbName, sessionNum=147)
    return tableHJ1, tableHJ2, tableHJ3

def mergeBehaviorTables(tableHJ1, tableHJ2, tableHJ3):
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
