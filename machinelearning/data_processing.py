import pandas as pd
import numpy as np
import sklearn
import random

class Data_processor():
    def __init__(self, file = 'listings.csv'):
        self.file = file
        self.df = pd.read_csv(file)

    def data_processing(self):
        df = self.df
        #select useful columns
        df2 = df[['id', 'host_since', 'host_response_time', \
                  'host_response_rate', 'host_is_superhost', \
                  'host_verifications', 'smart_location', 'latitude', 'longitude', \
                  'is_location_exact', 'property_type', 'room_type', 'accommodates', \
                  'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
                  'square_feet', 'price', 'security_deposit', 'cleaning_fee',
                  'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
                  'number_of_reviews', 'review_scores_rating', 'cancellation_policy']]

        # Convert column values and fill Nan values
        df2['price'] = df2['price'].apply(lambda x: x[1:-3])
        df2['price'] = df2['price'].str.replace(',', '')
        df2['price'] = df2['price'].apply(pd.to_numeric)
        df2['extra_people'] = df2['extra_people'].fillna('$0.00')
        df2['extra_people'] = df2['extra_people'].apply(lambda x: x[1:-3])
        df2['extra_people'] = df2['extra_people'].str.replace(',', '')
        df2['extra_people'] = df2['extra_people'].apply(pd.to_numeric)
        df2['cleaning_fee'] = df2['cleaning_fee'].fillna('$0.00')
        df2['cleaning_fee'] = df2['cleaning_fee'].apply(lambda x: x[1:-3])
        df2['cleaning_fee'] = df2['cleaning_fee'].str.replace(',', '')
        df2['cleaning_fee'] = df2['cleaning_fee'].apply(pd.to_numeric)
        df2['security_deposit'] = df2['security_deposit'].fillna('$0.00')
        df2['security_deposit'] = df2['security_deposit'].apply(lambda x: x[1:-3])
        df2['security_deposit'] = df2['security_deposit'].str.replace(',', '')
        df2['security_deposit'] = df2['security_deposit'].apply(pd.to_numeric)
        # df2['accommodates'] = df2['accommodates'].apply(pd.to_numeric)
        df2['host_response_rate'] = df2['host_response_rate'].fillna('0%')
        df2['host_response_rate'] = df2['host_response_rate'].apply(lambda x: x[:-1])
        df2['host_response_rate'] = df2['host_response_rate'].apply(pd.to_numeric) / 100
        df2['host_since'] = df2['host_since'].str.replace('-', '').apply(pd.to_numeric)

        # Fill missing text type value with 'Unknown'
        text_cols = list(df2.select_dtypes(include=['object']).columns)
        for c in text_cols:
            df2[c] = df2[c].fillna('Unknown')

        # Drop outliers
        places = df2['smart_location'].value_counts().to_dict()
        total = 0
        #ALL = df2.shape[0]
        sub_list = []
        for key, item in places.items():
            total += item
            if item <= 20:
                break
            sub_list.append(key)
        # drop entries which has a uncommon geo location
        df5 = df2.loc[df2['smart_location'].isin(sub_list)]
        # drop entries which has unlike price
        df5 = df5.loc[(df5['price'] < 1000) & (df5['price'] >= 5)]
        # Save cleaned version of data
        self.df_cleaned = df5
        df5.to_csv('df_cleaned.csv')
        return True


    def data_to_numeric(self, numeric_fea=None):
        '''
        This function will take out numeric features from dataset and normalized it.
        :param: numeric_fea: A list of numeric features we need to use in our model
        :return: True
        '''
        df3 = self.df_cleaned.drop(['amenities', 'host_verifications', 'square_feet','id'], axis=1)
        text_cols = list(df3.select_dtypes(include=['object']).columns)
        df3 = df3.drop(text_cols, axis=1)
        df3 = df3.fillna(0)
        df3p = df3['price'].tolist()
        df3_cp = df3
        df3 = df3.drop(['price'], axis=1)

        from sklearn import preprocessing
        data_matrix = df3.values
        cols = list(df3.columns)
        normalized_data = preprocessing.scale(data_matrix, axis=0)
        dfsub = pd.DataFrame(normalized_data, columns=cols)
        dfsub['price'] = pd.Series(df3p)
        #drop unecessary cols:
        if numeric_fea:
            dfsub = dfsub[numeric_fea]
        else:
            dfsub = dfsub[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', \
                       'minimum_nights', 'maximum_nights', 'longitude', 'latitude']]
            df3 = df3[['accommodates', 'bathrooms', 'bedrooms', 'beds', \
                       'minimum_nights', 'maximum_nights', 'longitude', 'latitude']]
            dfsub_unormalized = pd.DataFrame(df3_cp.values, columns=list(df3_cp.columns))

        #Save the numeric features
        self.df_numeric = dfsub
        dfsub.to_csv('df_numeric.csv')
        self.df_numeric_unormalized = dfsub_unormalized
        dfsub_unormalized.to_csv('df_numeric_unormalized.csv')
        return True


    def numeric_and_dummy(self, loc_only=True):
        '''
        This function will make a data set which concate numeric features and dummy features
        :param:loc_only: If loc is True(default) we will only use location dummy features, otherwise
                        we will include other useful dummy features
        :return: True
        '''
        dfsub = self.df_numeric
        df3 = self.df_cleaned.drop(['amenities', 'host_verifications', 'square_feet', 'id'], axis=1)
        if loc_only:
            dummy_df = pd.get_dummies(df3['smart_location'])
        else:
            pass

        npsub = np.concatenate((dfsub.values, dummy_df.values), axis=1)
        dfsub = pd.DataFrame(npsub, columns=list(dfsub.columns) + list(dummy_df.columns))
        dfsub = dfsub.fillna(0)
        self.df_dummy_numeric = dfsub
        return True



    def make_train_test(self, dummy=False, unormalized = True):
        '''
        This function will make the train_test dataset for training
        :param dummy: If dummy set as False(default) we are only gonna use numeric features
        :return: True
        '''
        if not dummy and not unormalized:
            dfsub = self.df_numeric
        elif not dummy and unormalized:
            dfsub = self.df_numeric_unormalized
        elif dummy:
            dfsub = self.df_dummy_numeric

        ind_all = set(range(dfsub.shape[0]))
        df_tr = dfsub.sample(frac=0.8)
        frac_set = set(df_tr.index)
        rest_set = ind_all - frac_set
        rest_ind = list(rest_set)
        print(df_tr.shape)
        print(dfsub.shape)
        y_train = pd.DataFrame(dfsub.iloc[df_tr.index, :]['price'])
        df_test = dfsub.iloc[rest_ind, :]
        y_test = pd.DataFrame(dfsub.iloc[rest_ind, :]['price'])
        #keep all the train test dataset
        self.X_train = df_tr
        df_tr.to_csv('df_tr.csv')
        self.Y_train = y_train
        print(df_tr.shape)
        print(y_train.shape)
        y_train.to_csv('y_train.csv')
        self.X_test = df_test
        df_test.to_csv('df_test.csv')
        self.Y_test = y_test
        y_test.to_csv('y_test.csv')

        return  True

    def data_embedding(self, fea):
        '''
        This function will embed the data into high dimension space and perform nearest neighbour clustering
        :param fea: features used for embedding
        :return: A embedded dataset with id as reference
        '''
        pass

if __name__ == '__main__':
    dp = Data_processor()
    dp.data_processing()
    dp.data_to_numeric()
    dp.make_train_test()

