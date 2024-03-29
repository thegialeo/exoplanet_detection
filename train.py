import os
import  argparse
from itertools import islice
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn, data
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from tsaug import RandomTimeWarp, RandomJitter
from scipy.fftpack import fft
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from mxnet.gluon.model_zoo import vision



def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def window_sliding(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def mx_window_sliding(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = list(islice(it, n))
    mx_result = nd.array(result[0]).expand_dims(0)
    for i in range(1, n):
        temp = nd.array(result[i]).expand_dims(0)
        mx_result = nd.concat(mx_result, temp, dim=0)
    if len(mx_result) == n:
        yield mx_result
    for elem in it:
        mx_result = nd.concat(mx_result[1:], nd.array(elem).expand_dims(0), dim=0)
        yield mx_result


def evaluate_accuracy(data_iter, net, ctx, win_size=None):
    """Evaluate accuracy of a model on the given data set."""
    with autograd.predict_mode():
        acc = mx.metric.Accuracy()
        if win_size is not None:
            loader = tqdm(data_iter)
        else:
            loader = data_iter
        for data, label in loader:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = net(data)
            acc.update(preds=output, labels=label)
    return acc.get()[1]


def evaluate_f1_score(data_iter, net, ctx, win_size=None):
    """Evaluate f1 score of a model on the given data set."""
    with autograd.predict_mode():
        f1_score = mx.metric.F1(average='macro')
        if win_size is not None:
            loader = tqdm(data_iter)
        else:
            loader = data_iter
        for data, label in loader:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = net(data)
            f1_score.update(preds=output, labels=label)
    return f1_score.get()[1]

def evaluate_ws_accuracy(data_iter, net, ctx, win_size):
    """Evaluate accuracy of a model on the given data set using window sliding."""
    with autograd.predict_mode():
        acc = mx.metric.Accuracy()
        for data, label in tqdm(data_iter):
            label = label.as_in_context(ctx)
            # window sliding over testset
            seq = nd.transpose(data)
            generator = mx_window_sliding(seq, win_size)
            data_slide = next(generator).expand_dims(0).as_in_context(ctx)
            for win in generator:
                data_slide = nd.concat(data_slide, win.expand_dims(0).as_in_context(ctx), dim=0)
            # compute mean network output over all windows of sequence
            output_sum = nd.zeros((data.shape[0], 2), ctx=ctx)
            for data in data_slide:
                data_processed = nd.array(preprocessing(nd.transpose(data).asnumpy())).as_in_context(ctx)
                output = net(data_processed)
                output_sum += output
            output_mean = output_sum / data_slide.shape[0]
            acc.update(preds=output_mean, labels=label)
    return acc.get()[1]


def evaluate_ws_f1_score(data_iter, net, ctx, win_size):
    """Evaluate f1 score of a model on the given data set using window sliding."""
    with autograd.predict_mode():
        f1_score = mx.metric.F1(average='macro')
        for data, label in tqdm(data_iter):
            label = label.as_in_context(ctx)
            # window sliding over testset
            seq = nd.transpose(data)
            generator = mx_window_sliding(seq, win_size)
            data_slide = next(generator).expand_dims(0).as_in_context(ctx)
            for win in generator:
                data_slide = nd.concat(data_slide, win.expand_dims(0).as_in_context(ctx), dim=0)
            # compute mean network output over all windows of sequence
            output_sum = nd.zeros((data.shape[0], 2), ctx=ctx)
            for data in data_slide:
                data_processed = nd.array(preprocessing(nd.transpose(data).asnumpy())).as_in_context(ctx)
                output = net(data_processed)
                output_sum += output
            output_mean = output_sum / data_slide.shape[0]
            f1_score.update(preds=output_mean, labels=label)
    return f1_score.get()[1]

def preprocessing(X, fourier=True, smoothing=True):
    # center data to mean
    mean = np.mean(X, axis=1).reshape(-1, 1)
    X_centered = X - mean
    # standard deviation
    std = np.std(X_centered, axis=1).reshape(-1, 1)
    # outlier removal (based on std)
    outlier = np.argwhere((X_centered - 5*std) > 0)
    for i in range(len(outlier)):
        X_centered[outlier[i, 0], outlier[i, 1]] = 0
    # standardization
    X_final = X_centered / std
    # fourier transform
    if fourier:
        X_final = fft(X_final, axis=1)
        X_final = X_final[:, 0: len(X_final[0]) // 2]
        X_final = np.abs(X_final)
    # gaussian smoothing
    if smoothing:
        X_final = gaussian_filter(X_final, 3)
    return X_final


def train(net, trainloader, testloader, criterion, trainer, ctx, batch_size, num_epochs, oversample=None, win_size=None,
          preprocess=True, fourier=True, smoothing=True, extra_aug=None, subfolder_name=None, method=None):
    """Train and evaluate a model with CPU or GPU."""
    
    print('Training on:', ctx)
    
    loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    train_f1_hist = []
    test_f1_hist = []
    
    for epoch in range(num_epochs):
        start = time.time()
        if win_size is not None:
            print("Training:")
            loader = tqdm(trainloader)
        else:
            loader = trainloader
        for i, (data, label) in enumerate(loader):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record(train_mode=True):
                output = net(data)
                loss = criterion(output, label)
            loss.backward()
            trainer.step(batch_size)

            # record loss
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - 0.01) * moving_loss + 0.01 * curr_loss)

        if win_size is not None:
            if epoch == (num_epochs-1):
                print("Evaluate Test Accuracy:")
                test_acc = evaluate_ws_accuracy(testloader, net, ctx, win_size)
                print("Evaluate Test F1 Score:")
                test_f1 = evaluate_ws_f1_score(testloader, net, ctx, win_size)
        else:
            test_acc = evaluate_accuracy(testloader, net, ctx, win_size)
            test_f1 = evaluate_f1_score(testloader, net, ctx, win_size)
        if win_size is not None:
            print("Evaluate Train Accuracy:")
        train_acc = evaluate_accuracy(trainloader, net, ctx, win_size)
        if win_size is not None:
            print("Evaluate Train F1 Score:")
        train_f1 = evaluate_f1_score(trainloader, net, ctx, win_size)
        if (win_size is not None) and (epoch < (num_epochs-1)):
            print('epoch {}, loss {:.5f}, train acc {:.3f}, train f1 {:.3f},  time {:.1f} sec'.format(
                epoch + 1, moving_loss, train_acc, train_f1, time.time() - start))
        else:
            print('epoch {}, loss {:.5f}, train acc {:.3f}, test acc {:.3f}, train f1 {:.3f}, test f1 {:.3f},  time {:.1f} sec'.format(
                epoch + 1, moving_loss, train_acc, test_acc, train_f1, test_f1, time.time() - start))


        loss_hist.append(moving_loss)
        train_acc_hist.append(train_acc)
        train_f1_hist.append(train_f1)
        if not ((win_size is not None) and (epoch < (num_epochs-1))):
            test_acc_hist.append(test_acc)
            test_f1_hist.append(test_f1)

    # create directory
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists(os.path.join('./plots', subfolder_name)):
        os.mkdir(os.path.join('./plots', subfolder_name))
    if not os.path.exists(os.path.join('./models', subfolder_name)):
        os.mkdir(os.path.join('./models', subfolder_name))
    if not os.path.exists(os.path.join('./logs', subfolder_name)):
        os.mkdir(os.path.join('./logs', subfolder_name))

    # ending string name
    end_path = "_" + method
    if oversample is not None:
        end_path += '_{}'.format(oversample)
    if win_size is not None:
        end_path += '_ws_{}'.format(win_size)
    if not preprocess:
        end_path += '_no_preprocessing'
    if not fourier:
        end_path += '_no_fourier'
    if not smoothing:
        end_path += '_no_smoothing'
    if extra_aug is not None:
        end_path += '_{}'.format(extra_aug)

    # save loss plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(loss_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('average loss', fontsize=14)
    if subfolder_name is not None:
        plt.savefig(os.path.join('./plots', subfolder_name, 'loss' + end_path + '.png'))
    else:
        plt.savefig(os.path.join('./plots', 'loss' + end_path + '.png'))

    # save train accuracy plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(train_acc_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    if subfolder_name is not None:
        plt.savefig(os.path.join('./plots', subfolder_name, 'train_accuracy' + end_path + '.png'))
    else:
        plt.savefig(os.path.join('./plots', 'train_accuracy' + end_path + '.png'))

    # save test accuracy plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(test_acc_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    if subfolder_name is not None:
        plt.savefig(os.path.join('./plots', subfolder_name, 'test_accuracy' + end_path + '.png'))
    else:
        plt.savefig(os.path.join('./plots', 'test_accuracy' + end_path + '.png'))

    # save train f1 score plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(train_f1_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('f1 score', fontsize=14)
    if subfolder_name is not None:
        plt.savefig(os.path.join('./plots', subfolder_name, 'train_f1_score' + end_path + '.png'))
    else:
        plt.savefig(os.path.join('./plots', 'train_f1_score' + end_path + '.png'))

    # save train f1 score plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(test_f1_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('f1 score', fontsize=14)
    if subfolder_name is not None:
        plt.savefig(os.path.join('./plots', subfolder_name, 'test_f1_score' + end_path + '.png'))
    else:
        plt.savefig(os.path.join('./plots', 'test_f1_score' + end_path + '.png'))

    # close all figures
    plt.close("all")

    # save model
    if subfolder_name is not None:
        net.save_parameters(os.path.join('./models', subfolder_name, "net" + end_path + '.params'))
    else:
        net.save_parameters(os.path.join('./models', "net" + end_path + '.params'))

    # save logs
    if subfolder_name is not None:
        file = open(os.path.join('./logs', subfolder_name, 'log' + end_path + '.txt'), 'w')
    else:
        file = open(os.path.join('./logs', 'log' + end_path + '.txt'), 'w')
    print('Final Loss:', loss_hist[-1], file=file)
    print('Final Train Accuracy:', train_acc_hist[-1], file=file)
    print('Final Test Accuracy:', test_acc_hist[-1], file=file)
    print('Final Train F1 Score:', train_f1_hist[-1], file=file)
    print('Final Test F1 Score:', test_f1_hist[-1], file=file)

    return [loss_hist[-1], train_acc_hist[-1], test_acc_hist[-1], train_f1_hist[-1], test_f1_hist[-1]]




def run_experiment(x_train, y_train, x_test, y_test, window_size, extra_aug, preprocess, fourier, smoothing, oversample, batch_size, 
                   num_workers, steps_epochs, lr, num_epochs, subfolder_name, ctx, method):
    """Run the whole experiment procedure"""

    print()
    print(80 * '-')
    print()

    # apply window sliding
    if window_size is not None:
        x_train_slide = []
        y_train_slide = []
        step_size = window_size // 2
        for i, seq in enumerate(x_train):
            for j, win in enumerate(window_sliding(seq, window_size)):
                if j % step_size == 0:
                    x_train_slide.append(win)
                    y_train_slide.append(y_train[i])
        x_train = np.array(x_train_slide)
        y_train = np.array(y_train_slide)
        print("Perform window sliding with size: {}".format(window_size))

    # apply extra augmentation: random time warp, random noise
    if extra_aug is not None:
        augmenter = (RandomTimeWarp() @ extra_aug
                     + RandomJitter() @ extra_aug)
        print("Perform Random Time Warp")
        print("Add Random Noise")

        x_train = augmenter.run(x_train)

    # pre-processing
    if preprocess:
        print("Pre-processing data")
        x_train = preprocessing(x_train, fourier, smoothing)
        if window_size is None:
            x_test = preprocessing(x_test, fourier, smoothing)
        print("Outlier Removal")
        print("Normalization")
        if fourier:
            print("Perform Fourier Transform")
        if smoothing:
            print("Perform Gaussian Smoothing")


    # oversampling
    if oversample is not None:
        if oversample == 'random':
            oversampler = RandomOverSampler(sampling_strategy=1.0)
        elif oversample == 'SMOTE':
            oversampler = SMOTE(sampling_strategy=1.0, k_neighbors=15)
        elif oversample == 'BorderSMOTE':
            oversampler =  BorderlineSMOTE(sampling_strategy=1.0, k_neighbors=15, m_neighbors=5)
        elif oversample == 'SVMSMOTE':
            oversampler = SVMSMOTE(sampling_strategy=1.0, k_neighbors=15, m_neighbors=8)
        elif oversample == 'ADASYN':
            oversampler = ADASYN(sampling_strategy=1.0, n_neighbors=8)
        else:
            print("{} is not a valid oversampling argument.".format(oversample))
            raise ValueError
        x_train, y_train = oversampler.fit_resample(x_train, y_train)
        print("Perform Oversampling: {}".format(oversample))

    if method == 'MLP' or method == 'CNN':
        # add channel dimension for CNN
        if method == 'CNN':
            x_train = np.expand_dims(x_train, axis=1)
            x_test = np.expand_dims(x_test, axis=1)

        # dataloader
        trainset = data.ArrayDataset(x_train, y_train)
        testset = data.ArrayDataset(x_test, y_test)

        trainloader = data.DataLoader(trainset, batch_size, True, num_workers=num_workers, pin_memory=True)
        testloader = data.DataLoader(testset, batch_size, True)

        # model
        if method == 'MLP':
            net = nn.Sequential()
            with net.name_scope():
                net.add(nn.Dense(100, activation='relu'),
                        nn.Dense(100, activation='relu'),
                        nn.BatchNorm(),
                        nn.Dropout(0.2),
                        nn.Dense(2))

        if method == 'CNN': 
            net = nn.Sequential()
            with net.name_scope():
                net.add(nn.Conv1D(channels=16, kernel_size=5, activation='relu'),
                        nn.MaxPool1D(pool_size=2, strides=2),
                        nn.Conv1D(channels=32, kernel_size=4, activation='relu'),
                        nn.MaxPool1D(pool_size=2, strides=2),
                        nn.Flatten(),
                        nn.Dense(2))

        # init weights
        net.collect_params().initialize(init=init.Xavier(), ctx=ctx)

        # scheduler + trainer
        it_per_epoch = math.ceil(len(trainset) / batch_size)
        steps_iterations = [s * it_per_epoch for s in steps_epochs]
        schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.5)
        optimizer = mx.optimizer.Adam(learning_rate=lr, lr_scheduler=schedule)
        criterion = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(params=net.collect_params(), optimizer=optimizer)

        # training
        print()
        scores = train(net, trainloader, testloader, criterion, trainer, ctx, batch_size, num_epochs, oversample,
                    window_size, preprocess, fourier, smoothing, extra_aug, subfolder_name, method)
        print()

        return scores

    elif method == 'KNN':
        # K-nearest neighbors classifier
        clf = KNeighborsClassifier()
        
        # training
        print("Train K-nearest Neighbors Classifier")
        clf.fit(x_train, y_train)
        
        # prediction
        pred = clf.predict(x_test)
        
        # evaluate test scores
        print("Test K-nearest Neighbors Classifier")
        f1 = f1_score(y_test, pred, average='macro')
        acc = accuracy_score(y_test, pred)

        return [acc, f1]
    
    elif method == 'SVC':
        # Support Vector Classifier
        clf = SVC()

        # training
        print("Train Support Vector Classifier")
        clf.fit(x_train, y_train)

        # prediction
        pred = clf.predict(x_test)

        # evaluate test scores
        print("Test Support Vector Classifier")
        f1 = f1_score(y_test, pred, average='macro')
        acc = accuracy_score(y_test, pred)

        return [acc, f1]

    elif method == 'RandomForest':
        # Random Forest Classifier
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        
        # training 
        print("Train Random Forest Classifier")
        clf.fit(x_train, y_train)

        # prediction
        print("Test Random Forest Classifier")
        pred = clf.predict(x_test)

        # evaluate test
        f1 = f1_score(y_test, pred, average='macro')
        acc = accuracy_score(y_test, pred)

        return [acc, f1]
    
    else:
        print("method {} can not be identified or is not implemented".format(method))
    
    


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--oversampling", dest='oversample', action='store',
                        help="Oversampling Method to apply: random, SMOTE, ADASYN")
    parser.add_argument("--window_sliding", dest='window_size', action='store', type=int,
                        help="Use window sliding for data augmentation")
    parser.add_argument("--no_preprocessing", dest="preprocess", action='store_false',
                        help="Disable pre-processing on dataset")
    parser.add_argument("--no_fourier_transform", dest="fourier", action='store_false',
                        help="Disable fourier transform on data")
    parser.add_argument("--no_gaussian_smoothing", dest="smoothing", action='store_false',
                        help="Disable gaussian smoothing")
    parser.add_argument("--extra_augmentation", dest='extra_aug', action='store', type=float,
                        help="Apply extra augmentation with given probability: Random Time Wrap and Random Noise")
    parser.add_argument("--num_epoch", dest='num_epochs', action='store', type=int,
                        help="Number of Epochs")
    parser.add_argument("--learning_rate", dest='lr', action='store', type=float,
                        help="Learning rate")
    parser.add_argument("--adaptive_learning", dest='steps_epochs', action='store', nargs='+', type=int,
                        help="Epochs at which learning rate will be reduced by factor of 10")
    parser.add_argument("--batch_size", dest='batch_size', action='store', type=int,
                        help="Batch size for training")
    parser.add_argument("--num_workers", dest='num_workers', action='store', type=int,
                        help='Number of workers for dataloader')
    parser.add_argument("--cross_validation", dest="cross_valid", action='store_true',
                        help="Run experiment with k-fold cross-validation")
    parser.add_argument("--method", dest="method", action='store',
                        help="Use another machine learning method than MLP")
    parser.add_argument("--cpu", dest="cpu", action='store_true',
                        help="Use CPU even if GPU is available")

    parser.set_defaults(preprocess=True, fourier=True, smoothing=True, oversample=None, window_size=None,
                        extra_aug=None, num_epochs=100, lr=1e-2, steps_epochs=[20, 40, 55, 70, 80, 90, 95, 100], batch_size=1024,
                        num_workers=8, cross_valid=None, method='MLP', cpu=False)
    args = parser.parse_args()


    # check if GPU available
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = try_gpu()


    if args.cross_valid is not None:
        # merge Kaggle trainset and testset
        df_train = pd.read_csv(os.path.join('./dataset', 'exoTrain.csv'))
        df_test = pd.read_csv(os.path.join('./dataset', 'exoTest.csv'))
        df_all = pd.concat([df_train, df_test])
        print("Load data")
        
        # separate data and labels
        X = np.float32(df_all.values[:, 1:])
        y = np.float32(df_all.values[:, 0] - 1) 

        # record scores
        loss_hist = []
        train_acc_hist = []
        test_acc_hist = []
        train_f1_hist = []
        test_f1_hist = []

        # stratified K-Fold cross-validation
        print("Prepare stratified k-fold cross-validation")
        skf = StratifiedKFold(n_splits=10)
        for k, (train_index, test_index) in enumerate(skf.split(X, y)):
            x_train = X[train_index]
            x_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            # run experiment
            scores = run_experiment(x_train, y_train, x_test, y_test, args.window_size, args.extra_aug, args.preprocess, args.fourier, args.smoothing, 
                                    args.oversample, args.batch_size, args.num_workers, args.steps_epochs, args.lr, args.num_epochs, "k-fold-cv-{}".format(k), ctx, args.method)
          
            # record scores 
            if args.method == 'MLP' or args.method == 'CNN':
                loss_hist.append(scores[0])
                train_acc_hist.append(scores[1])
                test_acc_hist.append(scores[2])
                train_f1_hist.append(scores[3])
                test_f1_hist.append(scores[4])
            else:
                test_acc_hist.append(scores[0])
                test_f1_hist.append(scores[1])

        # compute mean
        if args.method == 'MLP' or args.method == 'CNN':
            mean_train_acc = sum(train_acc_hist) / len(train_acc_hist)
            mean_train_f1 = sum(train_f1_hist) / len(train_f1_hist)
        
        mean_test_acc = sum(test_acc_hist) / len(test_acc_hist)
        mean_test_f1 = sum(test_f1_hist) / len(test_f1_hist)

        # print final results
        print()
        print(80 * '#')
        print()
        print("k-fold cross-validation results:")
        if args.method == 'MLP':
            print("Training Accuracy:", train_acc_hist)
            print("Training F1 Score:", train_f1_hist)
        print("Test Accuracy:", test_acc_hist)
        print("Test F1 Score:", test_f1_hist)
        if args.method == 'MLP':
            print("Mean Training Accuracy:", mean_train_acc)
            print("Mean Training F1 Score:", mean_train_f1)
        print("Mean Test Accuracy:", mean_test_acc)
        print("Mean Test F1 Score:", mean_test_f1)

        # ending string name
        end_path = "_" + args.method
        if args.oversample is not None:
            end_path += '_{}'.format(args.oversample)
        if args.window_size is not None:
            end_path += '_ws_{}'.format(args.window_size)
        if not args.preprocess:
            end_path += '_no_preprocessing'
        if not args.fourier:
            end_path += '_no_fourier'
        if not args.smoothing:
            end_path += '_no_smoothing'
        if args.extra_aug is not None:
            end_path += '_aug_{}'.format(args.extra_aug)

        # save log file
        file = open(os.path.join('./logs', 'log_cross-validation' + end_path + '.txt'), 'w')
        if args.method == 'MLP':
            print("Training Accuracy:", train_acc_hist, file=file)
            print("Training F1 Score:", train_f1_hist, file=file)
        print("Test Accuracy:", test_acc_hist, file=file)
        print("Test F1 Score:", test_f1_hist, file=file)
        if args.method == 'MLP':
            print("Mean Training Accuracy:", mean_train_acc, file=file)
            print("Mean Training F1 Score:", mean_train_f1, file=file)
        print("Mean Test Accuracy:", mean_test_acc, file=file)
        print("Mean Test F1 Score:", mean_test_f1, file=file)


    else:
        # load data according to Kaggle train-test split
        df_train = pd.read_csv(os.path.join('./dataset', 'exoTrain.csv'))
        df_test = pd.read_csv(os.path.join('./dataset', 'exoTest.csv'))

        x_train = np.float32(df_train.values[:, 1:])
        y_train = np.float32(df_train.values[:, 0] - 1)
        x_test = np.float32(df_test.values[:, 1:])
        y_test = np.float32(df_test.values[:, 0] - 1)
        print("Load data")

        # run experiment
        run_experiment(x_train, y_train, x_test, y_test, args.window_size, args.extra_aug, args.preprocess, args.fourier, 
                    args.smoothing, args.oversample, args.batch_size, args.num_workers, args.steps_epochs, args.lr, args.num_epochs, "Kaggle-split", ctx, args.method)