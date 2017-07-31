import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pylab import *
from matplotlib import gridspec

import mne
from mne import create_info, find_events, Epochs, pick_types
from mne.time_frequency import tfr_multitaper

from scipy.fftpack import fft
from scipy import signal
from scipy.signal import butter, cheby2, filtfilt, resample

import couchdb

def addOffset(data):
    # Used to plot multiple lines on the same plot with a regular offset
    newData = (10*data/np.mean(data)) + 2*np.arange(data.shape[1]-1,-1,-1)
    return newData

def plotDataSubset(data, stim, startSound, t0=0, t1=1, fs=2048.):
    # Subset
    start = int(np.round(fs*t0))
    end = int(np.round(fs*t1))
    subData = data[start:end]
    # spread lines equally on the y-axis
    subData = addOffset(subData)
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
    for i in startSound[0]:
        if (i>start and i<end):
            print 'Sound starts at %f seconds (red vertical line)' % (i/fs)
            plt.axvline(i/fs)
    plt.legend(subData.columns, bbox_to_anchor=(1, 1), ncol=4)

    plt.subplot(gs[1])
    plt.plot(stim)
    plt.legend(stim.columns, bbox_to_anchor=(1, 1), ncol=4)
    plt.show()
    plt.close()

def removeElectrodes(data, electrodes):
    data1 = data.drop(electrodes, axis=1)
    return data1

def getEvents(raw, eventCode):
    # Returns events from triggers in raw data with MNE methods
    events = mne.find_events(raw)
    eventsDf = pd.DataFrame(events)

    # Keep only event corresponding to the beginning of the trial (code: event_code)
    startEvents = eventsDf.loc[eventsDf[2]==eventCode]

    startEvents = startEvents.set_index([np.arange(len(startEvents))])
    startEvents.index.name = 'start'
    return startEvents

def getTrialData(data, events, trialNum=0, electrode=None, baselineDur=0.1,
        trialDur=None, fs=2048.):
    # baselineDur and duration are in seconds
    baselineDurSamples = int(np.round(baselineDur * fs))

    start = events[0][trialNum]-baselineDurSamples
    if (trialDur is None):
        if (trialNum<events[0].shape[0]-1):
            end = events[0][trialNum+1]
        else:
            # last trial = 10s
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



def plotSpec(y, freqMin=None, freqMax=None, yMin=None, yMax=None, fs=2048.):
    Y = fft(y)
    Y = pd.Series(Y.real)

    N = len(Y)
    Ts = 1.0/fs; # sampling interval
    t = np.arange(0, N/fs, Ts) # time vector

    tf = np.linspace(0.0, 1.0/(2.0*Ts), N//2)

    plt.figure
    plt.plot(tf, 2.0/N * np.abs(Y[0:N//2]),'r')
    if (freqMin is not None):
        plt.xlim(left=freqMin)
    if (freqMax is not None):
        plt.xlim(right=freqMax)
    if (yMin is not None):
        plt.ylim(bottom=yMin)
    if (yMax is not None):
        plt.ylim(top=yMax)
    plt.xlabel('Freq (Hz)')
    plt.show()
    plt.close()

# def meanElectrodes(data, trialNum, baselineDur, trialDur, events, fs=2048.):
#     allElectrodes = pd.DataFrame()
#     for electrode in data.columns:
#         dataElectrode = getTrialData(data, events=events,
#             trialNum=trialNum, electrode=electrode, baselineDur=baselineDur,
#             trialDur=trialDur, fs=fs)
#         dataElectrodeNormalised = normalizeFromBaseline(dataElectrode,
#             baselineDur=baselineDur, fs=fs)
#         allElectrodes[electrode] = pd.Series(dataElectrodeNormalised)
#     mean = allElectrodes.mean(axis=1)
#     return mean

# def meanFFTElectrodes(data, events, trialNum, baselineDur=0.1, fs=2048.):
#     # Create df to store fft of each electrodes
#     allFFT = pd.DataFrame()
#     for electrode in data.columns:
#         dataElectrode = getTrialData(data, events=events,
#             trialNum=trialNum, electrode=electrode, fs=fs)
#         meanDataElectrode = normalizeFromBaseline(dataElectrode,
#             baselineDur=baselineDur, fs=fs)
#         Y = fft(meanDataElectrode)
#         # print Y[:10], '\naaaaa', dataElectrode[:10]
#         allFFT[electrode] = pd.Series(Y.real)
#     mean = allFFT.mean(axis=1)
#     return mean, allFFT

def meanElectrodePerTrial(data, trialNumList, events, trialDur,
        baselineDur=0.1,  fs=2048.):
    allTrials = pd.DataFrame()
    for trialNum in trialNumList:
        mean = meanElectrodes(data, trialNum=trialNum, baselineDur=baselineDur,
                              trialDur=trialDur, events=events, fs=fs)
        mean = mean.reset_index(drop=True)
        # meanNorm = normalizeFromBaseline(mean, baselineDur=baselineDur)
        allTrials['trial%d' % trialNum] = pd.Series(mean)

    # convert baselineDur in frames
    start = int(np.round(baselineDur *fs))

    # Change index to have x-axis in seconds
    allTrials = allTrials.set_index(np.arange(-start, allTrials.shape[0]-start)/fs)
    meanTrials = allTrials.mean(axis=1)
    return meanTrials, allTrials

def getTrialsAverage(data, events, trialDur=None, trialNumList=None, baselineDur=0.1,
        normalize=False, startOffset=None, fs=2048.):
    # Calculate average across trials for this Series
    if (trialNumList is None):
        trialNumList = [events.shape[0]]

    allTrials = pd.DataFrame()
    for trialNum in trialNumList:
        trialData = getTrialData(data, events=events, trialNum=trialNum,
            baselineDur=baselineDur, trialDur=trialDur, fs=fs)
        if normalize:
            trialData = normalizeFromBaseline(trialData,
                baselineDur=baselineDur, fs=fs)
        if startOffset is not None:
            startOffsetSamples = int(np.round(startOffset*fs))
            trialData = trialData[startOffsetSamples:]
            # print trialData
        trialData = trialData.reset_index(drop=True)
        allTrials['trial%d' % trialNum] = pd.Series(trialData)

    # convert baselineDur in frames
    start = int(np.round(baselineDur *fs))

    # Change index to have x-axis in seconds
    allTrials = allTrials.set_index(np.arange(-start, allTrials.shape[0]-start)/fs)

    meanTrials = allTrials.mean(axis=1)
    return meanTrials, allTrials

# def plotERP(data, trialNumList, events, trialDur, fs=2048., baselineDur=0.1):
#     print 'Average over %d trials' % len(trialNumList)
#     meanTrials, allTrials = meanElectrodePerTrial(data, trialNumList=trialNumList,
#         trialDur=trialDur, baselineDur=baselineDur, events=events, fs=fs)

#     plt.figure
#     plt.plot(meanTrials)
#     plt.axvline(x=0, color='grey', linestyle='dotted')
#     plt.axvspan(-baselineDur, 0, alpha=0.3, color='grey')
#     plt.xlabel('Time (s)')
#     plt.show()
#     plt.close()

def plotERPElectrodes(data, trialNumList, events, trialDur=None, fs=2048.,
        baselineDur=0.1, electrodes='Fp1', normalize=False, facet=False,
        startOffset=None):
    print 'Average of %d trials' % len(trialNumList)
    meanTrials = pd.DataFrame()
    for electrode in electrodes:
        meanTrials[electrode], allTrials = getTrialsAverage(data=data[electrode], events=events,
            trialDur=trialDur, trialNumList=trialNumList, baselineDur=baselineDur,
            normalize=normalize, startOffset=startOffset, fs=fs)

    if (facet):
        print 'Faceting...'
        meanTrials.plot(subplots=True)
    else:
        plt.figure
        plt.plot(meanTrials)
        plt.axvline(x=0, color='grey', linestyle='dotted')
        plt.axvspan(-baselineDur, 0, alpha=0.3, color='grey')
        plt.xlabel('Time (s)')
        plt.legend(meanTrials.columns, bbox_to_anchor=(1, 1), ncol=4)
        plt.show()
        plt.close()

def plotFFT(meanTrials, average=False, freqMin=None, freqMax=None, yMin=None,
    yMax=None, fs=2048.):
    N = meanTrials.shape[0]
    Ts = 1.0/fs; # sampling interval
    t = np.arange(0, N/fs, Ts) # time vector
    tf = np.linspace(0.0, 1.0/(2.0*Ts), N//2)
    newMeanTrials = 2.0/N * np.abs(meanTrials.iloc[0:N//2, :])
    newMeanTrials = newMeanTrials.set_index(tf)
    if (average):
        plt.figure()
        plt.plot(newMeanTrials)
        if (freqMin is not None):
            plt.xlim(left=freqMin)
        if (freqMax is not None):
            plt.xlim(right=freqMax)
        if (yMin is not None):
            plt.ylim(bottom=yMin)
        if (yMax is not None):
            plt.ylim(top=yMax)
        plt.legend(meanTrials.columns, bbox_to_anchor=(1, 1), ncol=4)
        plt.show()
        plt.close()
    else:
        newMeanTrials.plot(ylim=(yMin, yMax), xlim=(freqMin, freqMax), subplots=True)

def plotFFTElectrodes(data, trialNumList, events, trialDur, fs=2048.,
        baselineDur=0.1, electrodes='Fp1', normalize=False, average=False,
        startOffset=None, freqMin=None, freqMax=None, yMin=None, yMax=None):
    print 'Average of %d trials' % len(trialNumList)
    meanTrials = pd.DataFrame()
    for electrode in electrodes:
        meanDataElectrode, allTrials = getTrialsAverage(data=data[electrode], events=events,
            trialDur=trialDur, trialNumList=trialNumList, baselineDur=baselineDur,
            normalize=normalize, startOffset=startOffset, fs=fs)
        Y = fft(meanDataElectrode)
        meanTrials[electrode] = pd.Series(Y.real)

    plotFFT(meanTrials, average=average, freqMin=freqMin, freqMax=freqMax,
        yMin=yMin, yMax=yMax, fs=fs)

def calculateBaseline(data, baselineDur=0.1, fs=2048.):
    # duration for the baseline calculation is in seconds
    durSamples = int(np.round(baselineDur*fs))
    subData = data[:durSamples]
    baseline = subData.mean()
    return baseline

def normalizeFromBaseline(data, baselineDur=0.1, fs=2048.):
    start = int(np.round(baselineDur*fs))
    baseline = calculateBaseline(data, baselineDur=baselineDur, fs=fs)
    normalized = data-baseline
    return normalized

def getBehaviorData(dbAddress, dbName, sessionNum):
    """
    Fetch behavior data from couchdb (SOA, SNR, trial duration...).

    Parameters
    ----------
    sessionNum : int
        If None table argument has to be provided and will be used as behavior
        data. Otherwise behavior data will be fetched from this sessionNum.

    Returns
    -------
    lookupTable : instance of pandas.core.DataFrame
        A dataframe containing trial data.
    """
    couch = couchdb.Server(dbAddress)
    db = couch[dbName]
    lookupTable = pd.DataFrame(columns=['trialNum', 'SOA', 'SNR', 'trialDur',
        'soundStart'])
    count = 0
    for docid in db.view('_all_docs'):
        if (docid['id'].startswith('infMask_%d' % sessionNum)):
            count += 1
            doc = pd.DataFrame(db.get(docid['id']))
            trialNum = int(docid['id'].split('_')[-1])
            toneCloudLen = doc.toneCloudParam.shape[0]
            trialDur = doc.trialDur[0]
            toneCloud = pd.DataFrame(db.get(docid['id'])['toneCloudParam'])
            # print toneCloud
            soundStart = np.min(toneCloud['time'])

            if (toneCloudLen < 700):
                recoverTargetSOA = 4
            elif ((toneCloudLen > 800) & (toneCloudLen < 1100)):
                recoverTargetSOA = 7
            elif (toneCloudLen > 1300):
                recoverTargetSOA = 13
            else:
                raise ValueError('check the value of toneCloudLen')

            SNR = doc.targetdB[0]
            lookupTable.loc[count] = pd.Series({'trialNum': trialNum,
                'SOA': recoverTargetSOA, 'SNR': SNR, 'trialDur': trialDur,
                'soundStart': soundStart})

    return lookupTable

def compareTimeBehaviorEEG(dbAddress, events, startSound, sessionNum=None,
    table=None, fs=2048.):
    """
    Check that the events got from triggers are the right duration with comparing
    to trial durations from behavioral data

    Parameters
    ----------
    events : instance of pandas.core.DataFrame
        First column has to contain the timing of events in frames.
    startSound : instance of pandas.core.DataFrame
        First column has to contain the timing of events in frames.
    sessionNum : int | None
        If None table argument has to be provided and will be used as behavior
        data. Otherwise behavior data will be fetched from this sessionNum.
    table: pandas DataFrame | None
        if None sessionNum will be used to fetch  behavior data otherwise
        behavior data is given directly in table.
    fs: float
        Sampling frequency to convert samples in duration (ms)

    Returns
    -------
    comparisonTable : instance of pandas.core.DataFrame
        A dataframe containing columns ['trialNum', 'trialDur_eeg (ms)',
        'trialDur_behavior (ms)', 'diff (ms)', 'soundStart (ms)',
        'soundStart_eeg (ms)', 'diffSound (ms)'] usable to assess the triggers
        reliability.
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
    for i in range(lookupTable.shape[0]-1):
        comparisonTable['trialNum'] = lookupTable['trialNum']
        # -2 because the inter trial duration is 2 seconds and 0.3 for the delay
        # for setting next trial
        trialDur_eeg = (eventsCopy[0][i+1] - eventsCopy[0][i]) - 2300
        trialDur_behavior = lookupTable['trialDur'][lookupTable['trialNum']==i].iloc[0]

        soundStart = lookupTable['soundStart'][lookupTable['trialNum']==i].iloc[0]*1000
        # If starting tone cloud is more than 100 ms, the target was the first
        # sound heard by the participant and it is played 100 ms after the start
        if (soundStart>100):
            soundStart = 100

        soundStart_eeg = startSoundCopy[0][i] - eventsCopy[0][i]

        diffSound = np.abs(soundStart_eeg - soundStart)

        diff = np.abs(trialDur_eeg - trialDur_behavior)
        comparisonTable.loc[count] = pd.Series({'trialDur_eeg': trialDur_eeg,
            'trialDur_behavior': trialDur_behavior, 'diff': diff,
            'soundStart': soundStart, 'soundStart_eeg': soundStart_eeg,
            'diffSound': diffSound})
        count += 1
    comparisonTable.columns = ['trialNum', 'trialDur_eeg (ms)',
        'trialDur_behavior (ms)', 'diff (ms)', 'soundStart (ms)',
        'soundStart_eeg (ms)', 'diffSound (ms)']
    return comparisonTable

def createDataFromRaw(raw):
    print 'convert raw to dataframe...'
    data = raw.to_data_frame()
    # print 'reset index...'
    # data = data.reset_index()
    return data

def loadEEG(name):
    if isinstance(name, list):
        temp = []
        for i in name:
            data = mne.io.read_raw_edf(i)
            temp.append(data)
        print temp
        raw = mne.concatenate_raws(temp)
    else:
        raw = mne.io.read_raw_edf(name)
    return raw

def discriminateEvents(events, threshold):
    # discriminate triggers on the same channel
    # calculate the rolling difference (between n and n+1)
    events['diff'] = events[0].diff()
    # replace the nan with the first value
    events['diff'].iloc[0] = events.iloc[0, 0]
    # select events with time distance superior to threshold
    events = events[events['diff']>threshold]
    events = events.reset_index(drop=True)
    del events['diff']
    return events

def getTrialNumList(table, SOA=None, SNR=None):
    if (SOA is not None and SNR is not None):
        print 'SOA: %s, SNR: %s' % (SOA, SNR)
        return table.trialNum[(table.SOA.isin(SOA)) & (table.SNR.isin(SNR))]
    if (SOA is None and SNR is None):
        print 'All SOA and all SNR'
        return table.trialNum
    if (SOA is not None):
        print 'SOA: %s, all SNR' % SOA
        return table.trialNum[table.SOA.isin(SOA)]
    if (SNR is not None):
        print 'SNR: %s, all SOA' % SNR
        return table.trialNum[table.SNR.isin(SNR)]

def doFFT(x):
    FFT = fft(x)
    return FFT.real

def plotFFTTrials(data, trialNumList, events, trialDur, baselineDur=0.1,
        startOffset=None, freqMin=None, freqMax=None, yMin=None, yMax=None,
        fs=2048., normalize=False, electrodes=['Fp1'], average=False):
    print 'Average of %d trials' % len(trialNumList)
    meanTrialsFFT = pd.DataFrame()
    for electrode in electrodes:
        meanTrials, allTrials = getTrialsAverage(data=data[electrode], events=events,
            trialDur=trialDur, trialNumList=trialNumList, baselineDur=baselineDur,
            normalize=normalize, startOffset=startOffset, fs=fs)
        allTrialsFFT = allTrials.apply(doFFT)
        meanTrialsFFT[electrode] = allTrialsFFT.mean(axis=1)

    print allTrials.shape, meanTrialsFFT.shape
    plotFFT(meanTrialsFFT, average=average, freqMin=freqMin, freqMax=freqMax,
        yMin=yMin, yMax=yMax, fs=fs)


def butterFilter(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def cheby2Filter(cutoff, bandType, fs, minAttenuation, order=5):
    nyq = 0.5 * fs
    cut = cutoff / nyq
    b, a = cheby2(order, minAttenuation, cut, btype=bandType)
    return b, a


def filterSignal(data, low, high, fs, order):
    b, a = butterFilter(low, high, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def chebyFilter(data, cutoff, bandType, order, minAttenuation=40, fs=2048.):
    b, a = cheby2Filter(cutoff=cutoff, bandType=bandType, fs=fs,
        minAttenuation=minAttenuation, order=order)
    print 'a and b: done...'
    y = signal.filtfilt(b, a, data)
    return y

def filterEachElectrode(data, cutoff, bandType, order):
    filteredData = pd.DataFrame()
    for electrode in data.columns:
        print 'Apply filter to electrode %s...' % electrode
        filteredData[electrode] = chebyFilter(data[electrode], cutoff=cutoff,
            bandType=bandType, order=order)
    print 'Done!'
    return filteredData

def downsample(data, oldFS, newFS):
    newNumSamples = int((data.shape[0] / oldFS) * newFS)
    newData = resample(data, newNumSamples)
    return pd.DataFrame(newData)

def downsampleEvents(events, oldFS, newFS):
    newEvents = events.copy()
    newEvents[0] = (events[0]/oldFS)*newFS
    newEvents[0] = newEvents[0].round(0).astype('int')
    return newEvents

def createChanFromEvents(events):
    newEvents = events.copy()
    newEvents[0] = newEvents[0].round(0).astype('int')
    chan = np.zeros(int(events.iloc[-1, 0]))
    chan[newEvents.iloc[:-2, 0]] = 8
    return pd.DataFrame(chan, columns = ['STI 014'])

def checkPlots(data1, data2, fs1, fs2, start, end):
    start1 = int(np.round(fs1*start))
    start2 = int(np.round(fs2*start))
    end1 = int(np.round(fs1*end))
    end2 = int(np.round(fs2*end))
    # take 2 seconds of signal from data1 for 1 electrode
    data1Sub = data1.iloc[start1:end1, 4:5]
    # take 2 seconds of signal from data2 for 1 electrode
    data2Sub = data2.iloc[start2:end2, 4:5]
    # x-axis in seconds
    x1 = np.arange(data1Sub.shape[0])/fs1
    x2 = np.arange(data2Sub.shape[0])/fs2

    plt.figure()
    plt.plot(x1, data1Sub)
    plt.plot(x2, data2Sub, alpha=0.7)
    plt.show()
    plt.close()

def refToMastoids(data, M1, M2):
    """
    Transform each electrode of data according to the average of M1 and M2

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
    newData = data.sub(mastoidsMean, axis=0)
    return newData


############## FUNCTIONS FOR THE PILOT SESSION ###################
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

def applyFilters(data):
    # Low pass
    dataFiltered = filterEachElectrode(data, cutoff=45, bandType='low', order=3)
    # high pass
    dataFiltered = filterEachElectrode(dataFiltered, cutoff=0.1, bandType='high', order=3)
    return dataFiltered
