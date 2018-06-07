import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import gridspec
import mne
from mne import find_events
from scipy.signal import resample, freqz_zpk, zpk2sos, sosfiltfilt, cheb2ord, iirdesign
import couchdb
import h5py
import time

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

    Returns:

    newData : instance of pandas.core.DataFrame
        The data with offset applied to each electrode.
    """

    newData = data + offset * np.arange(data.shape[1]-1,-1,-1)
    return newData

def applyDSS(data, dss):
    """
    Apply the electrodes weights obtained through the Denoising Source Separation
    (DSS) to the data matrix using dot product.

    Parameters
    ----------
    data : array-like
        2D matrix of shape (time, electrodes) or 3D matrix of shape
        (trials, time, electrodes).
    dss : array-like
        2D matrix of shape (electrodes, electrodes) resulting of the DSS computation.
        See output of the `computeDSS()` function for more details.

    Returns:

    weightedData : array-like
        2D matrix of shape (time, electrodes) or 3D matrix of shape
        (trials, time, electrodes) containing the input data weighted
        by the matrix dss.
    """
    if (np.ndim(data)==2):
        weightedData = np.dot(data, dss)
    elif (np.ndim(data)==3):
        trials = data.shape[0]
        time = data.shape[1]
        electrodes = data.shape[2]

        # Reshape data from 3D matrix of shape (trials, time, electrodes) to 2D matrix
        # of shape (time, electrodes)
        data2D = data.reshape((trials*time), electrodes)

        weightedData = np.dot(data2D, dss)
        # Reshape to reconstruct the 3D matrix
        weightedData = weightedData.reshape(trials, time, electrodes)
    else:
        print('data wrong dimensions')

    return weightedData

def calculateBaseline(data, baselineDur=0.1, fs=2048.):
    """
    Calculate and return the baseline (average of each data point) of a signal.
    The baseline will be calculated from the first `baselineDur` seconds of this
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

    Returns:

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

    Returns:

    zpk :

    filteredData : instance of numpy.array | instance of pandas.core.DataFrame
        The filtered data.
    """

    wp = [cutoff[1]/(fs/2), cutoff[2]/(fs/2)]
    ws = [cutoff[0]/(fs/2), cutoff[3]/(fs/2)]

    z, p, k = iirdesign(wp = wp, ws= ws, gstop=gstop, gpass=gpass,
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

    Returns:

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

    Returns:

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

def computeDSS(cov0, cov1):
    """
    Compute the Denoising Source Separation (DSS) from unbiased (cov0) and biased (cov1)
    covariance matrices.

    Parameters
    ----------
    cov0 : array-like
        Covariance matrix of unbiased data.
    cov1 : array-like
        Covariance matrix of biased data.

    Returns:

    DSS : array-like
        Matrix of shape identical to cov0 and cov1 containing the weights that can be
        applied on data.
    """
    # cov0 is the unbiased covariance and cov1 the biased covariance
    P, D = PCAFromCov(cov0)
    D = np.abs(D)

    # whiten
    N = np.diag(np.sqrt(1./D))
    c2 = N.T.dot(P.T).dot(cov1).dot(P).dot(N)

    Q, eigenVals2 = PCAFromCov(c2)
    eigenVals2 = np.abs(eigenVals2)

    W = P.dot(N).dot(Q)
    N2=np.diag(W.T.dot(cov0).dot(W))
    W=W.dot(np.diag(1./sqrt(N2)))
    return W

def computeFFT(data, fs):
    """
    Compute the FFT of `data` and return also the axis in Hz for further plot.

    Parameters
    ----------
    data : array
        First dataframe.
    fs : float
        Sampling frequency in Hz.

    Returns:

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
        Matrix of shape (time, electrode).
    pickFreq : float
        Frequency in Hz of the pick for which we want to calculate the relative energy.
    showPlot : boolean
        A plot of the FFT can be shown.
    fs : float
        Sampling frequency in Hz.

    Returns:

    pickRatio : float
        Relative energy of the pick.
    """
    pickWidth = 1
    N = data.shape[0]
    # Calculate the FFT of `data`
    fAx, fftData = computeFFT(data, fs)

    # Convert pick from Hz to bin number
    pickBin = int(np.round(pickFreq*(N/fs)))

    # Extract power at the frequency bin
    pickData = fftData[pickBin:pickBin+pickWidth]
    # Average power across time
    pickDataMean = pickData.mean(axis=0)

    # Extract power around the frequency bin
    nonPickBin = np.concatenate([np.arange(pickBin-5, pickBin),
                                np.arange(pickBin+1, pickBin+5+1)]);

    nonPickData = fftData[nonPickBin]
    nonPickDataMean = nonPickData.mean(axis=0)

    pickRatio = pickDataMean / nonPickDataMean

    if (showPlot):
        plt.figure()
        plt.plot(fAx, fftData, linewidth=0.5)
        plt.show()

    return pickRatio

def covUnnorm(data):
    """
    Calculate the unnormalized covariance of the the matrix `data`. Covariance in numpy
    normalize by dividing the dot product by (N-1) where N is the number
    of samples. The sum across electrodes of the raw (not normalized)
    covariance is returned.

    Parameters
    ----------
    data : array-like
        3D matrix of shape (trials, time, electrodes).

    Returns:

    cov: array-like
        Covariance matrix of shape (electrodes, electrodes)
    """
    electrodeNum = data.shape[2]
    trialNum = data.shape[0]
    cov = np.zeros((electrodeNum, electrodeNum))
    for i in range(trialNum):
        cov += np.dot(data[i,:,:].T, data[i,:,:])
    return cov

def create3DMatrix(data, trialTable, events, trialList, trialDur, fs, normalize, baselineDur=0.1):
    """
    """
    trials = trialTable.copy()
    trials = trials[trials['trialNum'].isin(trialList)]
    totalTrialNum = np.max(trials['trialNum'])
    m = trials.shape[0]
    print m, totalTrialNum

    electrodeNumber = data.shape[1]
    trialSamples = int(np.round((trialDur+baselineDur)*fs))
    # number of features: each sample for each electrode
    n = int(np.round(trialDur*fs*electrodeNumber))
    # Get trial data
    X = np.zeros((m, trialSamples, electrodeNumber))

    print 'creating matrix of shape (trials=%d, time=%ds, electrodes=%d)' % (X.shape[0],
                                                                            X.shape[1]/fs,
                                                                            X.shape[2])
    count = 0
    for i in range(totalTrialNum+1):
        # Check if this trial is in our subset
        if (i in trialList.unique()):
            trial = getTrialDataNP(data.values, events=events,
                                   trialNum=i, baselineDur=baselineDur,
                                   startOffset=0,
                                   trialDur=trialDur, fs=fs)
            # Normalization
            if (normalize):
                trial = normalizeFromBaseline(trial,
                                baselineDur=baselineDur, fs=fs)

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

    Returns:

    stim : instance of pandas.core.series.Series
        Series containing the stimulus channel reconstructed from events.
    """

    newEvents = events.copy()
    newEvents[0] = newEvents[0].round(0).astype('int')
    chan = np.zeros(int(events.iloc[-1, 0]))
    chan[newEvents.iloc[:-2, 0]] = 8
    stim = pd.Series(chan, columns = ['STI 014'])
    return stim

def crossValidate(data, dataBiased, trialTable):
    """
    Compute DSS from all trials except one and apply it one the one. Do that
    with all trials as cross validation.

    Parameters
    ----------
    data : array-like
        3D matrix of shape (trials, time, electrodes) containing unbiased data
    dataBiased : array-like
        3D matrix of shape (trials, time, electrodes) containing biased data
        (for instance band-pass filtered)

    Returns:

    allDSS : array-like
        Matrix of shape identical to dataBiased containing the weighted data
    """
    trials = data.shape[0]
    time = data.shape[1]
    electrodes = data.shape[2]

    trials4Hz = getTrialNumList(trialTable, noise=[False],
                                SOA=[4]).astype(int).values
    trials4HzNum = len(trials4Hz)

    allDSS = np.zeros((trials4HzNum, time, electrodes))
    for i in range(trials4HzNum-1):
        # All 4Hz trials except ith
        trialsToUse4Hz = np.delete(trials4Hz, i)
        # All trials except ith
        trialsToUseAll = np.concatenate([np.arange(0,trials4Hz[i]),
                                         np.arange(trials4Hz[i]+1, trials)])

        # Calculate covariance for unbiased and biased data
        cov0 = covUnnorm(data[trialsToUseAll,:,:])
        cov1 = covUnnorm(dataBiased[trials4Hz,:,:])

        DSS = computeDSS(cov0, cov1)

        # Test on ith trial
        XTest = data[trials4Hz[i],:,:]
        dataDSS = applyDSS(XTest, DSS)
        # Push ith trial
        allDSS[i,:,:] = dataDSS
    return allDSS

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

    Returns:

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

    Returns:

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

    Returns:

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
    data : array-like
        Data to resample.
    oldFS : float
        The sampling frequency of data.
    newFS : float
        The new sampling frequency.

    Returns:

    newData : instance of pandas.DataFrame
        The downsampled dataset.
    """

    newNumSamples = int((data.shape[0] / oldFS) * newFS)
    newData = resample(data, newNumSamples)
    return newData

def FFTTrials(data, events, trialNumList, baselineDur, trialDur, fs, normalize,
    electrodes):
    """
    """
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

def filterAndDownsampleByChunk(raw, fs, newFS, chunkNum=10):
    """
    Downsample data. Filters have to be used before downsampling. To be more
    efficient, the filters and downsampling are applied by chunk of data.

    Parameters
    ----------
    raw : instance of mne.io.edf.edf.RawEDF
        Raw data.
    fs : float
        The sampling frequency of data.
    newFS :
        The sampling frequency of data after downsampling.
    chunkNum : int
        Number of chunk used to process data.

    Returns:

    data : array-like
        The filtered and downsampled data.
    """
    # Calculate the number of sample per chunk
    subsetLen = int(np.round(len(raw)/float(chunkNum)))

    acc = {}
    for i in range(chunkNum):
        tic = time.time()
        print '...'*chunkNum

        start = subsetLen*i
        end = subsetLen*(i+1)-1

        # Take part of data
        eegData = raw[:, start:end][0].T

        zpk, eegData2Hz = chebyBandpassFilter(eegData, [1.8, 2., 30., 35.],
                                                 gstop=80, gpass=1, fs=fs)
        eegData2HzNewFS = downsampleNP(eegData2Hz, oldFS=fs, newFS=newFS)
        acc[i] = eegData2HzNewFS

        toc = time.time()
        print (str(1000*(toc-tic)))

    # Re concatenate processed data
    totalSampleNum = int(np.round(len(raw)/fs*newFS))
    totalChanNum = len(raw.ch_names)

    data = np.zeros((totalSampleNum, totalChanNum))

    subset = totalSampleNum/chunkNum
    for i in acc:
        start = subset*i
        end = subset*(i+1)
        data[start:end, :] = acc[i]
        print i, ' done!'
    return data

def getBehaviorData(dbAddress, dbName, sessionNum, recover=True):
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

    Returns:

    lookupTable : instance of pandas.core.DataFrame
        A dataframe containing trial data.
    """

    couch = couchdb.Server(dbAddress)
    db = couch[dbName]
    lookupTable = pd.DataFrame(columns=['trialNum', 'SOA', 'SNR', 'targetRate',
        'targetFreq','trialDur', 'soundStart', 'deviant', 'noise', 'target',
        'score'])
    count = 0
    for docid in db.view('_all_docs'):
        if (docid['id'].startswith('infMask_%d' % sessionNum)):
            count += 1

            trialNum = int(docid['id'].split('_')[-1])

            if (db.get(docid['id'])['toneCloudParam'] is not None and recover):
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
                # target SOA can be infered from the number of tones in the cloud
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
                target=doc.target
                score=doc.score


            lookupTable.loc[count] = pd.Series({'trialNum': trialNum,
                    'SOA': recoverTargetSOA, 'SNR': SNR, 'targetRate': targetRate,
                    'targetFreq': targetFreq, 'trialDur': trialDur,
                    'soundStart': soundStart, 'deviant': deviant,
                    'noise': recoverNoise, 'target': target, 'score': score})

    return lookupTable

def getEvents(raw, eventCode, shortest_event=None):
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

    Returns:

    startEvents : instance of pandas.core.DataFrame
        Dataframe containing the list of timing corresponding to the event code
        in the first column. The second column contains the code before the event
        and the third the code of the selected event.
    """
    if shortest_event:
      events = mne.find_events(raw, shortest_event=shortest_event)
    else:
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

    Returns:

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

    Returns:

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
    """
    See getTrialData
    """
    if (events[0][trialNum]==None):
        return
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

    Returns:

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
    """
    """
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

    Returns:

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

    Returns:

    normalized : instance of pandas.core.DataFrame
        The normalized data.
    """

    start = int(np.round(baselineDur*fs))
    baseline = calculateBaseline(data, baselineDur=baselineDur, fs=fs)
    normalized = data-baseline
    return normalized

def PCAFromCov(cov):
    """
    Get PCA components and eignvalues from covariance matrix.

    Parameters
    ----------
    cov : array-like
        Covariance matrix of shape (electrodes, electrodes).

    Returns:

    PCAComp : array-like
        Matrix of shape (electrodes, electrodes). Columns are the eigenvectors.
    eigenVals : array-like
        Matrix of shape (electrodes, 1).
    """

    # Calculate eigenvectors and eigenvalues
    eigenVals, eigenVecs = np.linalg.eig(cov)
    # Get the index of the sorted eigenvalues
    idx1 = np.argsort(eigenVals.T)[::-1]
    # Sort the eigenvalues in descending order
    eigenVals = np.sort(eigenVals.T)[::-1]
    # Reorder the columns of eigenvectors (the top components are the first columns)
    PCAComp = eigenVecs[:,idx1]
    return PCAComp, eigenVals

def plot3DMatrix(data, picks, trialList, average, trialDur, offset, normalize, fs):
    """
    """
    trialDurSamples = int(np.round(trialDur*fs))
    offsetSamples = int(np.round(offset*fs))
    baselineDur = 0.1
    baselineDurSamples = int(np.round(baselineDur*fs))
    # subset trials
    dataSub = data[trialList,:,:]

    # subset electrodes
    dataSub = dataSub[:,:,picks]

    # subset trialDur
    dataSub = dataSub[:, offsetSamples:trialDurSamples, :]
    print dataSub.shape

    # Normalization
    if (normalize):
        dataNorm = normalizeFromBaseline(dataSub,
                        baselineDur=baselineDur, fs=fs)

    # calculate mean across trials
    print 'Averaging %d trials...' % (dataNorm.shape[0])
    dataMean = np.mean(dataNorm, axis=0)

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
    Plot all electrodes from t0 to t1 with an y-axis offset. The stimulus channel is
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

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the data subset in the time domain.
    """

    # Subset
    start = int(np.round(fs*t0))
    end = int(np.round(fs*t1))
    subData = data[start:end]

    # Normalize between 0 and 1
    subData = (subData-subData.min())/(subData.max()-subData.min())

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

    Returns:

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

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the FFT.
    """

    N = data.shape[0]
    Ts = 1.0/fs; # sampling interval
    t = np.arange(0, N/fs, Ts) # time vector
    tf = np.linspace(0.0, 1.0/(2.0*Ts), N//2)
    newMeanTrials = 2.0/N * np.abs(data.iloc[0:N//2, :])
    newMeanTrials = newMeanTrials.set_index(tf)
    xticks = [4, 7, 12, 14, 16, 28, 36, 44, 72, 88, 100]
    if (facet):
        newMeanTrials.plot(ylim=(yMin, yMax), xlim=(freqMin, freqMax),subplots=True)
        plt.xlabel('frequency (Hz)')
        plt.xticks(xticks)
    else:
        plt.figure()
        plt.plot(newMeanTrials, linewidth=0.5)
        if (freqMin is not None):
            print freqMin
            plt.xlim(left=freqMin)
        if (freqMax is not None):
            plt.xlim(right=freqMax)
        if (yMin is not None):
            plt.ylim(bottom=yMin)
        if (yMax is not None):
            plt.ylim(top=yMax)
        # plt.legend(data.columns, bbox_to_anchor=(1, 1), ncol=4)
        plt.xlabel('frequency (Hz)')
        plt.xticks(xticks)
        # plt.show()

def plotFFTElectrodes(data, trialNumList, events, trialDur, fs,
    baselineDur=0.1, electrodes='Fp1', normalize=False, facet=False,
    freqMin=None, freqMax=None, yMin=None, yMax=None, startOffset=0,
    noiseAve=None, averageTimeFirst=False):
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
    averageTimeFirst : bool, default to False
        If True: average data in the time domain and then do the FFT.
        If False: do the FFT for each trial and then average in the frequency domain

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the FFT.
    """

    print 'Average of %d trials' % len(trialNumList)
    allTrials = pd.DataFrame()
    baselineDurSamples = int(np.round(baselineDur))
    if averageTimeFirst:
        for electrode in electrodes:
            allY = pd.DataFrame()
            for trialNum in trialNumList:
                trialData = getTrialData(data[electrode], events=events, trialNum=trialNum,
                    baselineDur=baselineDur, trialDur=trialDur, fs=fs,
                    startOffset=startOffset)
                if normalize:
                    trialData = normalizeFromBaseline(trialData,
                        baselineDur=baselineDur, fs=fs)

            trialData = trialData.reset_index(drop=True)

            allY[trialNum] = fftpack.fft(trialData[baselineDurSamples:])
            allTrials[electrode] = pd.Series(allY.mean(axis=1))
    else:
        for electrode in electrodes:
            meanDataElectrode, allTrialsElectrode = getTrialsAverage(data=data[electrode], events=events,
                trialDur=trialDur, trialNumList=trialNumList, baselineDur=baselineDur,
                normalize=normalize, fs=fs, startOffset=startOffset)
            Y = fftpack.fft(meanDataElectrode[baselineDurSamples:])
            allTrials[electrode] = pd.Series(np.abs(Y))

    plotFFT(allTrials, facet=facet, freqMin=freqMin, freqMax=freqMax,
        yMin=yMin, yMax=yMax, fs=fs)

def plotFFTNP(data, average, fs):
    """
    """
    if average:
        data = data.mean(axis=1)

    fAx, fftData = computeFFT(data, fs)

    plt.figure()
    plt.plot(fAx, fftData)
    plt.xlabel('frequency (Hz)')
    plt.xticks([4, 7, 13, 26])
    plt.xlim(0, 35)
    # plt.show()

def plotFilterResponse(zpk, fs):
    """
    Plot the filter frequency response.

    Parameters
    ----------
    zpk : array-like
        The 3 parameters of the filter [z, p, k].
    fs : float
        Sampling frequency in Hz.

    Returns:

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
    """
    """
    average = np.mean(data, axis=1)
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

    Returns:

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
    """
    """
    mastoidsMean = np.mean([M1, M2], axis=0)
    mastoidsMean = mastoidsMean.reshape(mastoidsMean.shape[0], 1)
    newData = data - mastoidsMean
    return newData
