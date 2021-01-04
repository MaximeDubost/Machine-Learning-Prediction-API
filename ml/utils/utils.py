import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings, sys, pickle
import datetime as dt
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataHandler:

    @classmethod
    def get_data(self):
        """ 
        Get data from GCS Bucket 
        """
        print('[1/2] Getting data... ', end='')
        df1 = pd.read_csv('https://storage.googleapis.com/h3-data/listings_final.csv', sep=';')
        df2 = pd.read_csv('https://storage.googleapis.com/h3-data/price_availability.csv', sep=';')
        print('Done.')
        return [df1, df2]
    
    @classmethod
    def get_group_data(self, data):
        """ 
        Merge both dataframes' data 
        """
        print('[2/2] Merging data... ', end='')
        result = pd.merge(data[0], data[1].groupby('listing_id').local_price.mean('local_price'), how='inner', on='listing_id')
        print('Done.')
        return result
    
    @classmethod
    def get_process_data(self):
        """ 
        Get & Merge data 
        """
        print("===| DataHandler |=== \n")
        result = self.get_group_data(self.get_data())
        return result


class FeatureRecipe:
    data = None
    variable_types = None
    useless_columns = []
    thresholded_columns = []
    duplicated_columns = []

    def __init__(self, data):
        """
        FeatureRecipe's Constructor
        """
        self.data = data
    
        
    def separate_variable_types(self):
        """ 
        Separate column variable types on lists 
        """
        print('[1/5] Separate variable types... ', end='')
        
        discreet, continues, boolean, categorical = [], [], [], []
        for column in self.data.columns:
            if self.data[column].dtype == np.dtype('int64'):
                discreet.append(self.data[column].name)
            elif self.data[column].dtype == np.dtype('float64'):
                continues.append(self.data[column].name)
            elif self.data[column].dtype == np.dtype('bool'):
                boolean.append(self.data[column].name)
            else:
                categorical.append(self.data[column].name)    
        self.variable_types = {"discreet": discreet, "continues": continues, "boolean": boolean, "categorical": categorical}

        print("Done.")
    
    
    def drop_uselessf(self):
        """ 
        Drop useless columns 
        """
        print('[2/5] Dropping useless features... ', end='')

        if "Unnamed: 0" in self.data.columns:
            self.useless_columns.append('Unnamed: 0')
        
        for column in self.data.columns:
            if self.data[column].isna().sum == len(self.data[column]):
                self.useless_columns.append(self.data[column].name)

        self.data.drop(columns=self.useless_columns, inplace=True)
            
        print("Done.")
        
        
    def deal_duplicate(self):
        """ 
        Drop duplicated columns 
        """
        print('[3/5] Dropping duplicates... ', end='')
        
        for col1_i in range(self.data.shape[1]):
            for col2_i in range(col1_i+1, self.data.shape[1]):
                if self.data.iloc[:, col1_i].equals(self.data.iloc[:, col2_i]):
                    self.duplicated_columns.append(self.data.iloc[:, col2_i].name)
        
        self.data.drop(columns=self.duplicated_columns, inplace=True)
        
        print("Done.")
          
    
    def drop_nanp(self, thresold: float):
        """ 
        Drop NaN columns according to a thresold 
        
        @params:
            - thresold: value between 0 included and 1 excluded
        """
        print("[4/5] Dropping NaN columns according to thresold (" + str(thresold) + ")... ", end='')
        
        for column in self.data.columns:
            if (self.data[column].isna().sum() / self.data.shape[0]) > thresold:
                self.thresholded_columns.append(column)
                
        self.data.drop(columns=self.thresholded_columns, inplace=True)
        
        print("Done.")
        
        
    def deal_dtime(self):
        """ TODO : Traiter les DateTime """
        print('[5/5] Dealing DateTime... Not Implemented')
        pass

    
    def prepare_data(self, threshold: float):
        print("===| FeatureRecipe |=== \n")
        self.separate_variable_types()
        self.drop_uselessf()
        self.deal_duplicate()
        self.drop_nanp(threshold)
        self.deal_dtime()
        
        print("\n• Variable types :")
        for vtype in self.variable_types.keys():
            print("- " + str(vtype) + " : " + str(self.variable_types[vtype]))
        print("\n• Useless dropped columns (" + str(len(self.useless_columns)) + " column(s)) : " + str(self.useless_columns))
        print("\n• Duplicated dropped columns (" + str(len(self.duplicated_columns)) + " column(s)) : " + str(self.duplicated_columns))
        print("\n• Nan dropped columns by thresold (" + str(len(self.thresholded_columns)) + " column(s) for a thresold of " + str(threshold) + ") : " + str(self.thresholded_columns))
        print("\n• Deal DateTime : Not Implemented")
    

class FeatureExtractor:

    def __init__(self, data: pd.DataFrame, unused_columns:list):
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.data = data
        self.unused_columns = unused_columns
        
    def drop_unusedf(self):
        print('[1/2] Dropping unused columns... ', end='')
        for column in self.data.columns:
            if column in self.unused_columns:
                self.data.drop(columns=column, inplace=True)
        print('Done.')
    
    def split_data(self, test_size: float, random_state: int, target: str):
        print('[2/2] Splitting data... ', end='')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data.loc[:, self.data.columns != target], self.data[target], test_size=test_size, random_state=random_state)
        print('Done.')
        
    def extract(self):
        print("===| FeatureExtractor |=== \n")
        self.drop_unusedf()
        self.split_data(0.3, 42, 'local_price')
        return self.x_train, self.x_test, self.y_train, self.y_test


class ModelBuilder:
    """
        Class for train and print results of ml model 
    """
    def __init__(self, model_path: str = None, save: bool = None):
        self.path = model_path
        self.save = save
        self.reg = LinearRegression()
    
    def __repr__(self):
        pass
    
    def predict_test(self, X) -> np.ndarray:
        return self.reg.predict(X)
    
    def predict_from_dump(self, X) -> np.ndarray:
        pass
    
    def save_model(self, path:str):
        # with the format : 'model_{}_{}'.format(date)
        res = pickle.dumps(self.reg)
        pickle.dump(res, "{}/model_{}.joblib".format(self.path, dt.datetime.now()))
    
    def print_accuracy(self, X_test, y_test):
        print("Accuracy : {}%".format(self.reg.score(X_test, y_test) * 100))
    
    def load_model(self):
        try:
            # load model
            return pickle.load("{}/model_{}.joblib".format(self.path, dt.datetime.now()))
        except:
            print("File not found")

