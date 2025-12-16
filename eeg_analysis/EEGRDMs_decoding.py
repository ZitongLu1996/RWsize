from neurora.decoding import tbyt_decoding_kfold
import numpy as np

def eegRDM_bydecoding(EEG_data, sub_opt=1, time_win=5, time_step=5, navg=5, time_opt="average", nfolds=5, nrepeats=2,
                      normalization=False):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) using classification-based neural decoding
    based on EEG-like data
    Parameters
    ----------
    EEG_data : array
        The EEG/MEG/fNIRS data.
        The shape of EEGdata must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    sub_opt: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
    time_win : int. Default is 5.
        Set a time-window for calculating the RDM for different time-points.
        Only when time_opt=1, time_win works.
        If time_win=5, that means each calculation process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.
        Only when time_opt=1, time_step works.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    nfolds : int. Default is 5.
        The number of folds.
        k should be at least 2.
    nrepeats : int. Default is 2.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    Returns
    -------
    RDM(s) : array
        The EEG/MEG/fNIR/other EEG-like RDM.
        If sub_opt=0, return int((n_ts-time_win)/time_step)+1 RDMs.
            The shape is [int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=1, return n_subs*int((n_ts-time_win)/time_step)+1 RDM.
            The shape is [n_subs, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
    Notes
    -----
    Sometimes, the numbers of trials under different conditions are not same. In NeuroRA, we recommend users to sample
    randomly from the trials under each conditions to keep the numbers of trials under different conditions same, and
    you can iterate multiple times.
    """

    if len(np.shape(EEG_data)) != 5:

        print("The shape of input for eegRDM() function must be [n_cons, n_subs, n_trials, n_chls, n_ts].\n")

        return "Invalid input!"

    # get the number of conditions, subjects, trials, channels and time points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    ts = int((ts - time_win) / time_step) + 1

    rdms = np.zeros([subs, ts, cons, cons])

    for con1 in range(cons):
        for con2 in range(cons):

            if con1 > con2:

                data = np.concatenate((EEG_data[con1], EEG_data[con2]), axis=1)
                labels = np.zeros([subs, 2*trials])
                labels[:, trials:] = 1
                rdms[:, :, con1, con2] = tbyt_decoding_kfold(data, labels, n=2, navg=navg, time_opt=time_opt,
                                                             time_win=time_win, time_step=time_step, nfolds=nfolds,
                                                             nrepeats=nrepeats, normalization=normalization,
                                                             pca=False, smooth=True)
                rdms[:, :, con2, con1] = rdms[:, :, con1, con2]
                print(con1, con2)

    if sub_opt == 0:

        return np.average(rdms, axis=0)

    else:

        return rdms

for sub in range(10):
    data = np.load('../eeg_test/sub-' + str(sub+1).zfill(2) + '/preprocessed_eeg_test.npy', allow_pickle=True).item()['preprocessed_eeg_data']
    # data: [nconditions * ntrials * nchannels * nts]c

    data = np.reshape(data, [200, 1, 80, 17, 100])

    eegrdms = eegRDM_bydecoding(data, sub_opt=1, time_win=1, time_step=1, nfolds=4, nrepeats=5)[0]

    np.save('RDMs/eegrdms_sub' + str(sub+1).zfill(2) + '.npy', eegrdms)
    print('EEG RDMs of Subject ' + str(sub+1).zfill(2))
"""eegdata = np.load('../eeg_test/sub-01/preprocessed_eeg_test.npy', allow_pickle=True).item()
data = eegdata['preprocessed_eeg_data']
print(data.shape)
subdata = np.zeros([1, 160, 17, 100])
subdata[0, :80] = data[0]
subdata[0, 80:] = data[1]
labels = np.zeros([1, 160], dtype=int)
labels[0, 80:] = 1

acc = tbyt_decoding_kfold(subdata, labels, n=2, navg=5, time_opt='average', time_win=1, time_step=1, nfolds=4, nrepeats=4, pca=False, smooth=True)[0]

import matplotlib.pyplot as plt

plt.plot(acc)
plt.show()"""