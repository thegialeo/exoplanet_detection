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
from sklearn.model_selection import KFold


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
          preprocess=True, fourier=True, smoothing=True, extra_aug=None, subfolder_name=None):
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
            with autograd.record():
                output = net(data)
                loss = criterion(output, label)
            loss.backward()
            trainer.step(batch_size)

            # record loss
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - 0.01) * moving_loss + 0.01 * curr_loss)

        if win_size is not None:
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
            print("Evaluate Test Accuracy:")
        train_f1 = evaluate_f1_score(trainloader, net, ctx, win_size)
        print('epoch {}, loss {:.5f}, train acc {:.3f}, test acc {:.3f}, train f1 {:.3f}, test f1 {:.3f}  time {:.1f} sec'.format(
            epoch + 1, moving_loss, train_acc, test_acc, train_f1, test_f1, time.time() - start))


        loss_hist.append(moving_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        train_f1_hist.append(train_f1)
        test_f1_hist.append(test_f1)

    # create directory
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    # ending string name
    end_path = ''
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




def run_experiment(x_train, y_train, x_test, y_test, window_size, extra_aug, preprocess, fourier, smoothing, oversample, batch_size, 
                   num_workers, steps_epochs, lr, num_epochs, subfolder_name):
    """Run the whole experiment procedure"""

    # check if GPU available
    ctx = try_gpu()
    
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


    # dataloader
    trainset = data.ArrayDataset(x_train, y_train)
    testset = data.ArrayDataset(x_test, y_test)

    trainloader = data.DataLoader(trainset, batch_size, True, num_workers=num_workers, pin_memory=True)
    testloader = data.DataLoader(testset, batch_size, True)


    # model
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(100, activation='relu'),
                nn.Dense(100, activation='relu'),
                nn.BatchNorm(),
                nn.Dropout(0.2),
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
    train(net, trainloader, testloader, criterion, trainer, ctx, batch_size, num_epochs, oversample,
          window_size, preprocess, fourier, smoothing, extra_aug, subfolder_name)




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
    parser.add_argument("--cross_validation", dest="cross_valid", action='store',
                        help="Run experiment with k-fold cross-validation")

    parser.set_defaults(preprocess=True, fourier=True, smoothing=True, oversample=None, window_size=None,
                        extra_aug=None, num_epochs=100, lr=1e-2, steps_epochs=[50, 80, 100], batch_size=1024,
                        num_workers=8, cross_valid=None)
    args = parser.parse_args()


    if args.cross_valid is not None:
        # merge Kaggle trainset and testset
        df_train = pd.read_csv(os.path.join('./dataset', 'exoTrain.csv'))
        df_test = pd.read_csv(os.path.join('./dataset', 'exoTest.csv'))
        df_all = pd.concat([df_train, df_test])
        
        # separate data and labels
        X = np.float32(df_all.values[:, 1:])
        y = np.float32(df_all.values[:, 0] - 1) 

        # K-Fold cross-validation k=5
        kf = KFold(n_splits=5)
        for k, (train_index, test_index) in enumerate(kf.split(X)):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            # run experiment
            run_experiment(x_train, y_train, x_test, y_test, args.window_size, args.extra_aug, args.preprocess, args.fourier, 
                           args.smoothing, args.oversample, args.batch_size, args.num_workers, args.steps_epochs, args.lr, args.num_epochs, "k-fold-cv-{}".format(k))

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
                       args.smoothing, args.oversample, args.batch_size, args.num_workers, args.steps_epochs, args.lr, args.num_epochs, "Kaggle-split")