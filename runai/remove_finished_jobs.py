#!/usr/bin/env python
import pandas as pd
import subprocess
from io import StringIO
from tqdm import tqdm


def main():
    jobs_result = subprocess.run(['runai', 'list', 'jobs'], stdout=subprocess.PIPE)
    jobs_output = StringIO(jobs_result.stdout.decode('utf-8'))
    df = pd.read_csv(jobs_output, sep='\\s{2,}', skiprows=1, engine='python')

    # Data columns (total 11 columns):
    #  #   Column                      Non-Null Count  Dtype
    # ---  ------                      --------------  -----
    #  0   NAME                        54 non-null     object
    #  1   STATUS                      54 non-null     object
    #  2   AGE                         54 non-null     object
    #  3   NODE                        54 non-null     object
    #  4   IMAGE                       54 non-null     object
    #  5   TYPE                        54 non-null     object
    #  6   PROJECT                     54 non-null     object
    #  7   USER                        54 non-null     object
    #  8   GPUs Allocated (Requested)  54 non-null     object
    #  9   PODs Running (Pending)      54 non-null     object
    #  10  SERVICE URL(S)              1 non-null      object
    # dtypes: object(11)

    df_filtered = df[
        df['STATUS'].isin(['Succeeded', 'Failed'])
    ]

    print("Deleting finished jobs:")
    print(df_filtered[['NAME', 'TYPE', 'STATUS']])

    # capture output of runai delete job <job_name>
    for job_name in tqdm(df_filtered['NAME']):
        result: subprocess.CompletedProcess = subprocess.run(
            ['runai', 'delete', 'job', job_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error deleting job {job_name}: {result.stderr.decode('utf-8')}")


if __name__ == '__main__':
    main()
