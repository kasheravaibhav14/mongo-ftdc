import bson
from sys import argv
import zlib
import struct
import io
from FTDC_analysis import FTDC_an
from datetime import datetime,timedelta
import ctypes
import pathlib
import os
import time

def int64(uint64_value):
    # Create a ctypes unsigned 64-bit integer from the input value
    uint64_type = ctypes.c_uint64(uint64_value)
    # Convert the unsigned 64-bit integer to a signed 64-bit integer
    int64_type = ctypes.c_int64(uint64_type.value)
    # Return the signed 64-bit integer value
    return int64_type.value

def convert_to_datetime(datetime_str):
    datetime_format = "%Y-%m-%dT%H-%M-%SZ-%f"
    datetime_obj = datetime.strptime(datetime_str, datetime_format)
    return datetime_obj

def getDTObj(date_string):
    format_string = "%Y-%m-%d_%H-%M-%S"
    parsed_datetime = datetime.strptime(date_string, format_string)
    return parsed_datetime

class FTDC:
    def __init__(self, metric_path,query_dt, outpath='',duration=600):
        self.fpath=metric_path
        self.metric_names=[]
        self.prev_metric_list=[]
        self.metric_list={}
        self.metaDocs=[]
        self.rawDataDocs=[]
        self.tdelta=timedelta(hours=3)
        self.qTstamp=query_dt
        self.outpath=outpath
        self.duration=duration

    def read_varuint(self,buf):
        value = 0
        shift = 0
        while True:
            b = buf.read(1)
            if not b:
                return -1
            byte = ord(b)
            value |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                break
        return value
    
    def __extract(self):
        for doc in self.items:
            if int(doc['type'])==1:
                self.rawDataDocs.append(doc)
            elif int(doc['type'])==0:
                self.metaDocs.append(dict(doc))
        
    def create_metric(self, data, prevkey=""):
        if isinstance(data, dict):
            for key,val in data.items():
                if prevkey=="":
                    nkey=key
                else:
                    nkey=prevkey+"."+key
                self.create_metric(val,nkey)
        elif isinstance(data, list):
            for idx,item in enumerate(data):
                nkey=prevkey+"._"+str(idx)
                self.create_metric(item,nkey)
        else:
            if type(data) == bson.Timestamp:
                k0=prevkey
                k1=prevkey+".inc"
                self.metric_names.append(k0)
                self.metric_names.append(k1)
                self.metric_list[k0]=[data.time*1000]
                self.metric_list[k1]=[data.inc]

            elif type(data) == datetime:
                ms_epoch=data.timestamp()*1000
                self.metric_names.append(prevkey)
                self.metric_list[prevkey]=[ms_epoch]

            elif type(data) != str and type(data) != bson.ObjectId:
                self.metric_names.append(prevkey)
                self.metric_list[prevkey]=[data]

    def parseBson(self,buf):
        size_data = buf.read(4)
        if len(size_data) < 4:
            return None  # Incomplete data, cannot form a valid BSON object

        size = struct.unpack("<i", size_data)[0]  # Unpack the size as a 32-bit signed integer
        data = size_data + buf.read(size - 4)  # Read the remaining bytes based on the size
        if len(data) < size:
            return None  # Incomplete data, cannot form a valid BSON object
        try:
            return bson.decode(data)  # Decode the BSON data
        except bson.InvalidBSON:
            return None  # Invalid BSON data
        
    def __decodeData(self):
        accumulate_metrics={}
        ndet_tot=0
        for doc in self.rawDataDocs:
            try:
                to_decode=doc['data']
                decompressed_dat = zlib.decompress(to_decode[4:]) #first 4 bytes = header(uncompressed by default)
                reader=io.BytesIO(decompressed_dat)
                res=self.parseBson(reader)
                if abs(res['start']-self.qTstamp) > self.tdelta:
                    continue
                self.create_metric(res)
                stats=reader.read(8)
                nmetrics,ndeltas=(struct.unpack("<I", stats[0:4]),struct.unpack("<I", stats[4:8]))
                ndet_tot+=(ndeltas[0]+1)
                nzeros=0
                for met_idx in range(nmetrics[0]):
                    base_val=self.metric_list[self.metric_names[met_idx]][0]
                    for del_idx in range(ndeltas[0]):
                        delta=0
                        if nzeros!=0:
                            delta=0
                            nzeros-=1
                        else:
                            delta= self.read_varuint(reader)
                            if delta==-1:
                                raise ValueError("Error in Reading")
                            if delta == 0:
                                nzeros= self.read_varuint(reader)
                                if nzeros==-1:
                                    raise ValueError("Error in Reading")
                        if type(base_val)==datetime:
                            base_val= base_val + timedelta(seconds=delta)
                        else:
                            base_val= base_val + int64(delta)
                        self.metric_list[self.metric_names[met_idx]].append(base_val)
                tstamp=datetime.fromtimestamp(self.metric_list['start'][0]/1000)
                tstamp=tstamp.strftime("%Y-%m-%d_%H-%M-%S")
                accumulate_metrics[tstamp]=self.metric_list
                self.metric_list={}
                self.metric_names=[]
                reader.close()
            except Exception as e:
                print("Failed to extract: ",e)
        print(ndet_tot)
        tstamp=(next(iter(accumulate_metrics)))
        an_obj=FTDC_an(accumulate_metrics,self.qTstamp,self.outpath,self.duration)
        an_obj.parseAll()

    def process(self):
        docs=[]
        for filepath in self.fpath:
            with open(filepath,'rb') as file:
                data=file.read()
            docs.extend(bson.decode_all(data))
        self.items = docs
        self.__extract()
        self.__decodeData()

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
    all_files= [i for i in files if i.name.startswith("metrics") and i.is_file()]
    all_files.sort()
    # print(all_files)
    for file in all_files:
        if "interim" not in file.name:
            # print(file.name)
            tstamp=convert_to_datetime(file.name[file.name.index('.')+1:])
            diff=abs(tstamp-query_dt)
            if diff <= timedelta(hours=6):
                # print(diff)
                filtered_files.append(file)
    filtered_files.sort()
    if all_files.index(filtered_files[-1]) == len(all_files)-2: # metrics.interim should be included
        filtered_files.append(all_files[-1])
    if len(filtered_files)==0:
        raise ValueError("No files corresponding to the queryTimestamp found. Please check the timestamp/path and try again!")
    # print(filtered_files)
    if len(argv)==5:
        duration=int(float(argv[4])*60)
        print("bucket duration set to: ",duration, "seconds")
        decoder = FTDC(filtered_files,query_dt,argv[3],duration)
    else:
        decoder = FTDC(filtered_files,query_dt,argv[3])
    decoder.process()
    st=time.time()-st
    print("Runtime in seconds: ",st)