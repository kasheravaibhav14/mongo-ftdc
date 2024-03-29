import pathlib
from sys import argv
from datetime import datetime,timedelta
from FTDC_decoder import FTDC
import time
import os 

def convert_to_datetime(datetime_str):
    datetime_format = "%Y-%m-%dT%H-%M-%SZ-%f"
    datetime_obj = datetime.strptime(datetime_str, datetime_format)
    return datetime_obj

def getDTObj(date_string):
    format_string = "%Y-%m-%d_%H-%M-%S"
    parsed_datetime = datetime.strptime(date_string, format_string)
    return parsed_datetime

if __name__ == "__main__":
    st=time.time()
    if(len(argv)<4):
        print("use python3 <FTDC_capture.py> <diagnostic.data> <qTstamp> <outfilename> <bucket duration in mins>")
        exit(1)
    KEY_NAME= 'OPENAI_API_KEY'
    key = os.environ.get(KEY_NAME)
    if key is None:
        print("Please ensure there is an API KEY in the environment variables under OPENAI_API_KEY")
        exit(1)
    dirPath=pathlib.Path(argv[1])
    if not dirPath.is_dir():
        raise("diagnostic.data path is invalid")
    try:
        query_dt=getDTObj(argv[2])
    except Exception as e:
        print("The queryTimestamp is not of the correct format: Y-m-d_H-M-S")
        exit(1)
    files = dirPath.glob("*")
    filtered_files=[]
    for file in files:
        if file.is_file() and "interim" not in file.name and file.name.startswith("metrics"):
            # print(file.name)
            tstamp=convert_to_datetime(file.name[file.name.index('.')+1:])
            diff=abs(tstamp-query_dt)
            if diff <= timedelta(hours=6):
                # print(diff)
                filtered_files.append(file)

    filtered_files.sort()
    if len(filtered_files)==0:
        raise ValueError("No files corresponding to the queryTimestamp found. Please check the timestamp/path and try again!")
    print(filtered_files)
    if len(argv)==5:
        duration=int(float(argv[4])*60)
        print("bucket duration set to: ",duration, "seconds")
        decoder = FTDC(filtered_files,query_dt,argv[3],duration)
    else:
        decoder = FTDC(filtered_files,query_dt,argv[3])
    decoder.process()
    st=time.time()-st
    print("Runtime in seconds: ",st)
