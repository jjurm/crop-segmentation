#!/usr/bin/env python
import argparse
import pandas as pd
import subprocess
from io import StringIO
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--dry_run', action='store_true', default=False, required=False,
                        help='Do not delete jobs, only print the list of jobs that would be deleted.')
    parser.add_argument('--status', nargs="+", default=['Succeeded', 'Failed', 'Deleted'], required=False, )
    return parser.parse_args()


def get_runai_jobs():
    jobs_result = subprocess.run(['runai', 'list', 'jobs'], stdout=subprocess.PIPE)
    jobs_output = StringIO(jobs_result.stdout.decode('utf-8'))
    df = pd.read_csv(jobs_output, sep='\\s{2,}', skiprows=1, engine='python')
    return df


def delete_jobs(df_filtered):
    for job_name in tqdm(df_filtered['NAME']):
        result: subprocess.CompletedProcess = subprocess.run(
            ['runai', 'delete', 'job', job_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error deleting job {job_name}: {result.stderr.decode('utf-8')}")


def main():
    args = parse_arguments()
    df = get_runai_jobs()

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
        df['STATUS'].isin(args.status)
    ]

    if not args.dry_run:
        print("Deleting finished jobs:")
    print(df_filtered[['NAME', 'TYPE', 'STATUS']])

    if not args.dry_run:
        delete_jobs(df_filtered)


if __name__ == '__main__':
    main()
