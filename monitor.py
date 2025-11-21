from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
import joblib
import pandas as pd 

def monitor():
    
    referece=pd.read_csv('data/Iris.csv').drop(columns=['Id','Species'])
    
    model=joblib.load('model/model.pkl')
    
    current_x=referece.sample(5)
    current_y=model.predict(current_x)
    
    current=current_x.copy()
    
    report=Report(metrics=[DataDriftPreset(drift_share=0.3)])
    report.run(reference_data=referece,current_data=current)
    report.save_html('model/drift_report.html')

if __name__=="__main__":
    monitor()