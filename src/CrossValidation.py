from sklearn import model_selection
import pandas as pd
"""
Binary Classificaiton
MultiClass Classification
MultiLabel Classification
Regression
MultiLabel Regression
HoldOut

"""
class CrossValidation:
    def __init__(self
    ,df
    ,target_cols
    ,shuffle
    ,problem_type='binary_classification'
    ,num_folds=5
    ,delimeter=','    
    ,random_state=42):
            
        self.dataframe = df
        self.dataframe['kfold'] = -1
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.delimeter = delimeter

        if self.shuffle is True:
            self.dataframe=self.dataframe.sample(frac=1,random_state=42).reset_index(drop=True)     

    def split(self):
        target = self.target_cols[0]
        
        if self.problem_type in ['binary_classification','multiclass_classification']:
            if self.num_targets != 1:
                raise Exception('Invalid Number of targets for this problem type')

            unique_values = len(self.dataframe[target].unique())        
            if unique_values==1:
                raise Exception("Only one unique value found")
            elif unique_values > 1:                
                kf=model_selection.StratifiedKFold(
                                                    n_splits = self.num_folds,
                                                    shuffle = False,
                                                    # random_state=self.random_state
                                                    )
                for fold,(train_idx,val_idx) in enumerate(kf.split(X=self.dataframe,y=self.dataframe[target].values)):
                    # print(len(train_idx),len(val_idx))
                    self.dataframe.loc[val_idx,'kfold']=fold
            
        if self.problem_type in ['regression','multiLabel_regression']:
            if self.num_targets != 1 and self.problem_type =='regression':
                raise Exception('Invalid Number of targets for this problem type')
            if self.num_targets < 2 and self.problem_type =='multiLabel_regression':
                raise Exception('Invalid Number of targets for this problem type')
            
            kf=model_selection.KFold(n_splits = self.num_folds                                                    
                                                    )
            for fold,(train_idx,val_idx) in enumerate(kf.split(X=self.dataframe)):
                    self.dataframe.loc[val_idx,'kfold']=fold

        elif self.problem_type.startswith('holdout_'):
            holdout_percentage=int(self.problem_type.split('_')[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe)-num_holdout_samples,'kfold'] = 0
            self.dataframe.loc[len(self.dataframe)-num_holdout_samples:,'kfold'] = 1

        if self.problem_type in ['multilabel_classification']:
            if self.num_targets != 1:
                raise Exception('Invalid Number of targets for this problem type')
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.delimeter)))
            print(targets)        
            kf=model_selection.StratifiedKFold(
                                                    n_splits = self.num_folds,
                                                    )
            for fold,(train_idx,val_idx) in enumerate(kf.split(X=self.dataframe,y=targets)):
                self.dataframe.loc[val_idx,'kfold']=fold
        
        
        else:
            raise Exception ('problem_type not understood')  


        return self.dataframe
if __name__=='__main__':
    df=pd.read_csv('../input/train_multilabel.csv')

    # cv=CrossValidation(df=df,target_cols=["target"],problem_type='binary_classification')
    
    # cv=CrossValidation(df=df,target_cols=["SalePrice"],problem_type='holdout_10')
    cv=CrossValidation(df=df,
                        target_cols=["attribute_ids"],
                        problem_type='multilabel_classification',
                        shuffle=True,
                        delimeter = ' ')

    
    df_split=cv.split()
    print(df_split.kfold.value_counts())

