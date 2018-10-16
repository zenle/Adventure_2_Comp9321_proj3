import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

def supervised_train(dummy = False):
    '''
    This function will train our dataset with a regression model.
    :param file: File name for the necessary dataset
    :param dummy: When it is False(defualt) we will only use numeric data for training, otherwise we use numeric and
            dummy variables.
    :return: A Regression model
    '''
    X_train = pd.read_csv('df_tr.csv', index_col=0)
    Y_train = pd.read_csv('y_train.csv', index_col=0).values
    print(X_train.shape)
    print(Y_train.shape)
    if dummy:
        rfr = RandomForestRegressor(n_estimators=2000, max_depth=10, random_state=0, \
                                    oob_score=True, n_jobs=-1)
    else:
        rfr = RandomForestRegressor(n_estimators=2000, max_depth=3, random_state=0, \
                                   oob_score=True, n_jobs=-1)
    rfr.fit(X_train, Y_train)
    print(rfr.oob_score_)
    return rfr

def unsupervised_learn(dummy = False):
    '''

    :param dummy:
    :return:
    '''
    pass


def price_predicting(input, learn_mode = 'mixed', dummy = False, train_mode = True, normalized = False):
    '''
    This function
    :param input: A data input with features in a dictionary format need to return a predicitng price
    :param learn_mode: This parameter take a string input. It can either be 'unsupervised', 'supervised'
            or 'mixed'(default). If it's 'unsupervised' this function will use clustering model and take average;
            if it's surpervised this function will use regression model and predict price; if it's mixed will take
            a weighted average price from both methods.
    :param dummy: When it is False(defualt) we will only use numeric data for training, otherwise we use numeric and
            dummy variables.
    :param train_mode: When this is True(default) we will train our model and save it. When learn_mode has been set as
                        unsupervised this train_mode will be ignored.
    :return: A predicting price
    '''
    if learn_mode == 'mixed' and train_mode:
        rfr = supervised_train()
        pickle.dump(rfr, open('RandomForrestRegressor.txt','w+'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            #TODO: Process unnormalized input to train an normalized data
            pass

        pred_up = unsupervised_learn(input)
        pred = pred_sup * 0.4 + pred_up * 0.6
        return pred
    elif learn_mode == 'mixed' and not train_mode:
        rfr = pickle.load(open('RandomForrestRegressor.txt', 'r'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            pass

        pred_up = unsupervised_learn(input)
        pred = pred_sup * 0.4 + pred_up * 0.6
        return pred

    elif learn_mode == 'supervised' and train_mode:
        rfr = supervised_train()
        pickle.dump(rfr, open('RandomForrestRegressor.txt', 'w+'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            # TODO: Process unnormalized input to train an normalized data
            pass
        return pred_sup

    elif learn_mode == 'supervised' and not train_mode:
        rfr = pickle.load(open('RandomForrestRegressor.txt', 'r'))
        x_test = pd.DataFrame(input)
        if not normalized:
            pred_sup = rfr.predict(x_test)
        else:
            pass

        return pred_sup

    elif learn_mode == 'unsupervised':
        pass



if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    rfr = supervised_train()
    df_test = pd.read_csv('df_test.csv', index_col=0)
    y_test = pd.read_csv('y_test.csv', index_col=0)
    y_pred = rfr.predict(df_test)
    me = np.sqrt(mean_squared_error(y_pred, y_test))
    print(me)




