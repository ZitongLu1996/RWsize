import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from neurora.stuff import show_progressbar, smooth_1d, smooth_2d

def ct_decoding_kfold(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5, nfolds=5, nrepeats=2,
                      normalization=False, pca=False, pca_components=0.95, smooth=True):

    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials, nchls, nts = np.shape(data)

    ncategories = np.zeros([nsubs], dtype=int)

    labels = np.array(labels)

    for sub in range(nsubs):

        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)

    if len(set(ncategories.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories[0]:

        print("\nThe number of categories for decoding doesn't match ncategories (" + str(ncategories) + ")!\n")

        return "Invalid input!"

    categories = list(sublabels_set)

    newnts = int((nts-time_win)/time_step)+1

    if time_opt == "average":

        avgt_data = np.zeros([nsubs, ntrials, nchls, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        acc = np.zeros([nsubs, newnts, newnts])

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        if normalization is True:
                            if pca is True:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                x_test = scaler.transform(xt[test_index])
                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(x_train)
                                x_test = Pca.transform(x_test)
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])

                            if pca is False:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(scaler.transform(xt[test_index]), y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                 y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                     y[test_index])

                        if normalization is False:
                            if pca is False:

                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(xt[train_index], y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(xt[test_index], y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(xtt[test_index], y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(xtt[test_index], y[test_index])

                            if pca is True:

                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(xt[train_index])
                                x_test = Pca.transform(xt[test_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(Pca.transform(xtt[test_index]), y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(Pca.transform(xtt[test_index]), y[test_index])

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 3))

    if time_opt == "features":

        avgt_data = np.zeros([nsubs, ntrials, nchls, time_win, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, :, t] = data[:, :, :, t * time_step:t * time_step + time_win]

        avgt_data = np.reshape(avgt_data, [nsubs, ntrials, nchls * time_win, newnts])

        acc = np.zeros([nsubs, newnts, newnts])

        total = nsubs * nrepeats * newnts * nfolds

        print("\nDecoding")

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls * time_win, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls * time_win, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls * time_win, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        if normalization is True:
                            if pca is True:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                x_test = scaler.transform(xt[test_index])
                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(x_train)
                                x_test = Pca.transform(x_test)
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(
                                            Pca.transform(scaler.transform(xtt[test_index])), y[test_index])

                            if pca is False:

                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(xt[train_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(scaler.transform(xt[test_index]), y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                 y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(scaler.transform(xtt[test_index]),
                                                                                     y[test_index])

                        if normalization is False:
                            if pca is False:

                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(xt[train_index], y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(xt[test_index], y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(xtt[test_index], y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(xtt[test_index], y[test_index])

                            if pca is True:

                                Pca = PCA(n_components=pca_components)
                                x_train = Pca.fit_transform(xt[train_index])
                                x_test = Pca.transform(xt[test_index])
                                svm = SVC(kernel='linear', tol=1e-4, probability=False)
                                svm.fit(x_train, y[train_index])
                                subacc[i, t, t, fold_index] = svm.score(x_test, y[test_index])

                                for tt in range(newnts - 1):
                                    if tt < t:
                                        xtt = x[:, :, tt]
                                        subacc[i, t, tt, fold_index] = svm.score(Pca.transform(xtt[test_index]),
                                                                                 y[test_index])
                                    if tt >= t:
                                        xtt = x[:, :, tt + 1]
                                        subacc[i, t, tt + 1, fold_index] = svm.score(Pca.transform(xtt[test_index]),
                                                                                     y[test_index])

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 3))

    if smooth is False:
        return acc

    if smooth is True:

        smooth_acc = smooth_2d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_2d(acc, n=smooth)

        return smooth_acc

def eegCTRDM_bydecoding(EEG_data, sub_opt=1, time_win=5, time_step=5, navg=5, time_opt="average", nfolds=5, nrepeats=2,
                      normalization=False):

    if len(np.shape(EEG_data)) != 5:

        print("The shape of input for eegRDM() function must be [n_cons, n_subs, n_trials, n_chls, n_ts].\n")

        return "Invalid input!"

    # get the number of conditions, subjects, trials, channels and time points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    ts = int((ts - time_win) / time_step) + 1

    ctrdms = np.zeros([subs, ts, ts, cons, cons])

    for con1 in range(cons):
        for con2 in range(cons):

            if con1 > con2:

                data = np.concatenate((EEG_data[con1], EEG_data[con2]), axis=1)
                labels = np.zeros([subs, 2*trials])
                labels[:, trials:] = 1
                ctrdms[:, :, :, con1, con2] = ct_decoding_kfold(data, labels, n=2, navg=navg, time_opt=time_opt,
                                                                time_win=time_win, time_step=time_step, nfolds=nfolds,
                                                                nrepeats=nrepeats, normalization=normalization,
                                                                pca=False, smooth=True)
                ctrdms[:, :, :, con2, con1] = ctrdms[:, :, :, con1, con2]
                print(ctrdms[:, :, :, con2, con1])
                print(con1, con2)

    if sub_opt == 0:

        return np.average(ctrdms, axis=0)

    else:

        return ctrdms

for sub in range(5):
    sub = sub + 5
    data = np.load('../eeg_test/sub-' + str(sub+1).zfill(2) + '/preprocessed_eeg_test.npy', allow_pickle=True).item()['preprocessed_eeg_data'][:, :, :, :, :70]
    # data: [nconditions * ntrials * nchannels * nts]

    data = np.reshape(data, [200, 1, 80, 17, 70])

    eegrdms = eegCTRDM_bydecoding(data, sub_opt=1, time_win=1, time_step=1, nfolds=4, nrepeats=5)[0]

    np.save('RDMs/eegctrdms_sub' + str(sub+1).zfill(2) + '.npy', eegrdms)
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