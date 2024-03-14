from sklearn import metrics
# from airflow import models as airflow_models
from airflow.models import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


#ML part with function.

def prepare_data():
    import pandas as pd
    print("inside  prepare data......")
    df= pd.read_csv("https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv")
    df= df.dropna()
    df.to_csv(f'final_df.csv',index=False)

def train_test_split():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    print("inside  split data......")
    final_data=pd.read_csv("final_df.csv")
    target_column= 'class'
    x=final_data.loc[:, final_data.columns != target_column]
    y=final_data.loc[:, final_data.columns == target_column]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)

    np.save(f'x_train.npy', x_train)
    np.save(f'x_test.npy', x_test)
    np.save(f'y_train.npy', y_train)
    np.save(f'y_test.npy', y_test)


def training_classifier():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x_train=np.load(f'x_train.npy',allow_pickle=True)
    y_train=np.load(f'y_train.npy',allow_pickle=True)

    classifier= LogisticRegression(max_iter=400)
    classifier.fit(x_train,y_train)
    import pickle
    with open(f'model.pkl', 'wb')as f:
        pickle.dump(classifier,f)

        print("train & saved pkl")

def predict_test():
    import pandas as pd
    import numpy as np
    import pickle
    print("inside  predict data......")
    with open(f'model.pkl', 'rb')as f:
        logistic_model=pickle.load(f)
    x_test= np.load(f'x_test.npy',allow_pickle=True)
    y_pred=logistic_model.predict(x_test)
    np.save(f'y_pred.npy',y_pred)

    print("print predict class......")
    print(y_pred)


def get_metrics():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_score

    print("inside metrics class......")

    y_test=np.load(f'y_test.npy',allow_pickle=True)
    y_pred=np.load(f'y_pred.npy',allow_pickle=True)
    acc=accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred,average='micro')
    print(metrics.classification_report(y_test,y_pred))

    print("\n model metricss:",{'Accuracy':round(acc,2),'precision':round(prec,2)})

with DAG(
    dag_id='ML_pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 3, 13),
    catchup=False
    
    ) as dag:


    task_prepare_data=PythonOperator(

        task_id='prepare_data',
        python_callable=prepare_data,
    )

    task_split_data=PythonOperator(

        task_id='train_test_split',
        python_callable=train_test_split,
    )
    task_train_data=PythonOperator(

        task_id='training_classifier',
        python_callable=training_classifier,
    )
    task_predict_data=PythonOperator(

        task_id='predict_test',
        python_callable=predict_test,
    )

    task_metrics_data=PythonOperator(

        task_id='get_metrics',
        python_callable=get_metrics,
    )

#flow
task_prepare_data >> task_split_data >> task_train_data >> task_predict_data >> task_metrics_data