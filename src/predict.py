from sklearn import preprocessing
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import os
from . import dispatcher
import joblib
import numpy as np



TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


def predict():
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)
    test_idx=test_df['id'].values  
    test_df = test_df.drop(['id'],axis=1)
    prediction=None  
    
    
    # label_encoders =  joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl"))
    # clf =  joblib.load(os.path.join(f"models",f"{MODEL}_{FOLD}_.pkl"))

    # test_df=test_df[train_df_columns]
    for FOLD in range(5):
        test_df = pd.read_csv(TEST_DATA)
        encoder=joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols=joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_columns.pkl"))
        for i in cols:
        
            lbl = encoder[i]
            # lbl=preprocessing.LabelEncoder()
            # lbl.fit(train_df[i].values.tolist()+valid_df[i].values.tolist() + test_df[i].values.tolist())
            test_df.loc[:,i]=lbl.transform(test_df[i].values.tolist())
            # valid_df.loc[:,i]=lbl.transform(valid_df[i].values.tolist())
            # label_encoders.append((i,lbl))   

        test_df=test_df[cols]
        clf=joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
        preds=clf.predict_proba(test_df)[:,1]
        print(preds)

        if FOLD==0:
            prediction=preds
        else:
            prediction+=preds
        
    prediction/=5
    print(prediction)
    subs=pd.DataFrame(np.column_stack((test_idx,prediction)),columns=['id','target'])
    subs['id']=subs['id'].astype(int)
    return subs          



if __name__=="__main__":
    submission=predict()
    submission
    submission.to_csv(f'models/{MODEL}_submission.csv',index=False)
        

        
