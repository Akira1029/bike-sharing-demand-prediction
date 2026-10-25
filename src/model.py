"""Reproducible calendar-only Bike Sharing baseline."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

FEATURES=['yr','mnth','season','hr','weekday','holiday','workingday']
def load_data(path):
    df=pd.read_csv(path)
    df['timestamp']=pd.to_datetime(df['dteday'])+pd.to_timedelta(df['hr'],unit='h')
    return df.sort_values('timestamp').reset_index(drop=True)
def make_model():
    return Pipeline([('prep',ColumnTransformer([('calendar',OneHotEncoder(handle_unknown='ignore',sparse_output=False),FEATURES)])),('model',RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=-1))])
def fit_and_evaluate(path):
    df=load_data(path); cut=int(len(df)*.85)
    train,test=df.iloc[:cut],df.iloc[cut:]
    model=make_model().fit(train[FEATURES],train['cnt'])
    predictions=model.predict(test[FEATURES]); y=test['cnt']
    metrics={'MAE':float(mean_absolute_error(y,predictions)),'RMSE':float(np.sqrt(mean_squared_error(y,predictions))),'R2':float(r2_score(y,predictions)),'n_test':len(test)}
    return model,metrics
if __name__=='__main__':
    import json
    model,metrics=fit_and_evaluate(Path(__file__).resolve().parents[1]/'data'/'hour.csv')
    print(json.dumps(metrics,indent=2))
