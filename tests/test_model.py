from pathlib import Path
from src.model import FEATURES,load_data,make_model

def test_data_integrity():
    df=load_data(Path(__file__).resolve().parents[1]/'data'/'hour.csv')
    assert len(df)==17379
    assert (df['cnt']==df['casual']+df['registered']).all()
    assert df['timestamp'].is_monotonic_increasing
    assert 'cnt' not in FEATURES and 'casual' not in FEATURES and 'registered' not in FEATURES

def test_model_predictions():
    df=load_data(Path(__file__).resolve().parents[1]/'data'/'hour.csv')
    model=make_model().fit(df.iloc[:100][FEATURES],df.iloc[:100]['cnt'])
    assert len(model.predict(df.iloc[100:105][FEATURES]))==5
