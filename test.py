import pandas as pd
from mxnet.gluon import nn, data
from train import *



if __name__ == "__main__":

    # check if GPU available
    ctx = try_gpu()

    # load data
    df_test = pd.read_csv(os.path.join('./dataset', 'exoTest.csv'))
    x_test = np.float32(df_test.values[:, 1:])
    y_test = np.float32(df_test.values[:, 0] - 1)
    print("Load data")

    # pre-processing
    x_test = preprocessing(x_test)
    print("Outlier Removal")
    print("Normalization")
    print("Perform Fourier Transform")
    print("Perform Gaussian Smoothing")

    # dataloader
    testset = data.ArrayDataset(x_test, y_test)
    testloader = data.DataLoader(testset, 1024, True)

    # model
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(100, activation='relu'),
                nn.Dense(100, activation='relu'),
                nn.BatchNorm(),
                nn.Dropout(0.2),
                nn.Dense(2))

    # load weights
    net.load_parameters("net_SMOTE.params", ctx=ctx)

    # evaluate
    test_acc = evaluate_accuracy(testloader, net, ctx)
    test_f1 = evaluate_f1_score(testloader, net, ctx)
    print("Test Accuracy:", test_acc)
    print("Test F1 Score:", test_f1)