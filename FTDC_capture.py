import pathlib
from sys import argv
from datetime import datetime,timedelta
from FTDC_decoder import FTDC
import time

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
    if(len(argv)!=3):
        print("use python3 <FTDC_capture.py> <diagnostic.data> <qTstamp>")
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
        if file.is_file() and "interim" not in file.name :
            # print(file.name)
            tstamp=convert_to_datetime(file.name[file.name.index('.')+1:])
            diff=abs(tstamp-query_dt)
            if diff < timedelta(hours=4,minutes=30):
                # print(diff)
                filtered_files.append(file)

    filtered_files.sort()
    print(filtered_files)
    decoder = FTDC(filtered_files,query_dt)
    decoder.process()
    st=time.time()-st
    print("Runtime in seconds: ",st)
