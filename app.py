"""Historical educational demo, NOT a live operational forecast."""
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st
from src.model import FEATURES,load_data,make_model

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Bike Sharing | Historical ML Demo',page_icon='🚲',layout='wide')
st.title('🚲 Bike Sharing Demand Prediction')
st.warning('Historical research demo (2011–2012). Educational/portfolio use only. NOT a live operational forecast.')
st.caption('Calendar-only random forest: no future weather or realized ridership inputs.')
@st.cache_resource(show_spinner='Training historical model…')
def train_model():
    df=load_data(ROOT/'data'/'hour.csv')
    cut=int(len(df)*.85)
    return make_model().fit(df.iloc[:cut][FEATURES],df.iloc[:cut]['cnt'])
model=train_model()
left,right=st.columns(2)
with left:
    day=st.date_input('Historical date',value=date(2012,10,1),min_value=date(2011,1,1),max_value=date(2012,12,31))
    hour=st.slider('Hour of day',0,23,8)
with right:
    holiday=st.checkbox('Holiday (dataset definition)',value=False)
    st.caption('Weekday, working-day and season fields are derived automatically. Holiday dates require manual confirmation against the original dataset definition.')
weekday=(day.weekday()+1)%7 # UCI: Sunday=0, Monday=1
workingday=int(weekday not in (0,6) and not holiday)
month=day.month
season=(month-1)//3+1 # Dataset season code: 1 Jan-Mar, 2 Apr-Jun, 3 Jul-Sep, 4 Oct-Dec
row=pd.DataFrame([{'yr':day.year-2011,'mnth':month,'season':season,'hr':hour,'weekday':weekday,'holiday':int(holiday),'workingday':workingday}])[FEATURES]
if st.button('Predict historical hourly demand',type='primary'):
    prediction=float(model.predict(row)[0]); st.metric('Estimated rentals / hour',f'{prediction:,.0f}')
    st.caption('Model output is a historical scenario estimate, not a 2026 prediction.')
st.divider()
st.subheader('Held-out historical test metrics')
c1,c2,c3=st.columns(3)
c1.metric('MAE','63.43');c2.metric('RMSE','98.18');c3.metric('R²','0.7917')
st.caption('Metrics are from the fixed chronological held-out test (last 15% of records); they are not recalculated from your inputs.')
st.subheader('Model limitations')
st.write('The model uses only calendar features, has no station-level data or weather forecasts, and has not been validated on contemporary data. Do not use for live fleet allocation.')
st.image(str(ROOT/'artifacts'/'shap_stability.png'),caption='Approximate SHAP across four historical subsamples (24 observations each).')
