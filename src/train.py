from sklearn import preprocessing
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import os
from . import dispatcher
import joblib



TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__=="__main__":
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)    
    
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    valid_df = valid_df.drop(['id','target','kfold'],axis=1)
    train_df = train_df.drop(['id','target','kfold'],axis=1)

    valid_df=valid_df[train_df.columns]

    label_encoders = {}
    
    for i in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[i].values.tolist()+valid_df[i].values.tolist() + test_df[i].values.tolist())
        train_df.loc[:,i] = lbl.transform(train_df[i].values.tolist())
        valid_df.loc[:,i] = lbl.transform(valid_df[i].values.tolist())
        label_encoders[i] = lbl        

    #Trainin Data
    clf= dispatcher.MODELS[MODEL]
    clf.fit(train_df,ytrain)
    preds=clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(yvalid,preds))

    joblib.dump(label_encoders,f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf,f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns,f"models/{MODEL}_{FOLD}_columns.pkl")   

    # print(preds)
