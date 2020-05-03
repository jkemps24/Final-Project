import torch
import pandas as pd

def import_csv(enddate):
    pruid=[48,59,46,13,10,61,12,62,35,11,24,47,60]
    df=pd.read_csv('covid19.csv')
    df=pd.DataFrame(df,columns =['pruid','date','numconf'])
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    torch_data = torch.zeros(13,80)
    for i in range (13):
        data=df[df.pruid.eq(pruid[i])]
        data.set_index('date', inplace=True)
        data.sort_index(inplace=True)
        idx = pd.date_range(start='01-31-2020',end=enddate,freq='D')
        data=data.reindex(idx)
        if data['numconf']['01-31-2020']!=data['numconf']['01-31-2020']:
            data['numconf']['01-31-2020']=0
        data=data.fillna(method='ffill')
        t = torch.tensor(data['numconf'].values)
        for j in range (80):
            torch_data[i,j]=t[j]
    return torch_data