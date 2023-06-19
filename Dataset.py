import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataSet:
    """
    Preprocessing.
    """
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocessing(self, horizon, back_horizon, train_size=0.7, val_size=0.2, scale=True):
        self.horizon = horizon
        self.back_horizon = back_horizon

        y = self.data.copy().astype('float')
        self.train = y[:int(train_size*len(y))]
        self.val = y[int(train_size*len(y))-self.back_horizon:int((train_size+val_size)*len(y))]
        self.test = y[int((train_size+val_size)*len(y))-self.back_horizon:]
        #TODO add scaling

        # Training set
        self.X_train, self.y_train = self.create_sequences_multi(self.train,
                                                                 self.train,
                                                                 self.horizon,
                                                                 self.back_horizon)

        # Validation set
        self.X_val, self.y_val = self.create_sequences_multi(self.val,
                                                             self.val,
                                                             self.horizon,
                                                             self.back_horizon)
        # Testing set
        self.X_test, self.y_test = self.create_sequences_multi(self.test,
                                                               self.test,
                                                               self.horizon,
                                                               self.back_horizon)

        # training on all database
        self.X_train_all, self.y_train_all = self.create_sequences_multi(y,
                                                                         y,
                                                                         self.horizon,
                                                                         self.back_horizon)


    @staticmethod
    def create_sequences_multi(X, y, horizon, time_steps):
        Xs, ys = [], []
        for i in range(0, len(X)-time_steps-horizon, 1):
            Xs.append(X[i:(i+time_steps), :])
            ys.append(y[(i+time_steps):(i+time_steps+horizon), :])
        return np.array(Xs), np.array(ys)
