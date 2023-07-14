import bson
import zlib
import struct
import io
from FTDC_analysis import FTDC_an
from datetime import datetime,timedelta
import ctypes
import os
import time
import argparse
from urllib.parse import urlparse
import string 
import secrets
import tarfile 
import requests 
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import shutil

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
    def __init__(self, metric_path,query_dt, outpath='',duration=600, exact=0):
        self.fpath=metric_path
        self.metric_names=[]
        self.prev_metric_list=[]
        self.metric_list={}
        self.metaDocs=[]
        self.rawDataDocs=[]
        self.tdelta=timedelta(hours=2, minutes=30)
        self.qTstamp=query_dt
        self.outpath=outpath
        self.duration=duration
        self.exact=exact

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
                if type(data) == bool:
                    data = int64(int(data))
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
        # print(ndet_tot)
        tstamp=(next(iter(accumulate_metrics)))
        an_obj=FTDC_an(accumulate_metrics,self.qTstamp,self.outpath,self.duration,self.exact)
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

def validate_directory(value):
    if not os.path.isdir(value) and not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"Directory '{value}' does not exist")
    return value

def validate_url(value):
    try:
        result = urlparse(value)
        if result.scheme and result.netloc:
            return value
        else:
            raise argparse.ArgumentTypeError('Invalid URL')
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid URL')

def validate_timestamp(value):
    # print(type(value))
    try:
        dtobj=datetime.fromtimestamp(int(value)//1000)
        return dtobj
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid timestamp, required format is YYYY-MM-DD HH:MM:SS')

def validate_interval(value):
    if not value.isdigit() or int(value) <= 0:
        raise argparse.ArgumentTypeError('Interval should be a positive integer')
    return int(value)

def find_files(directory, paths=[]):
    for entry in os.scandir(directory):
        if entry.is_file():
            paths.append(entry.path)
        elif entry.is_dir():
            find_files(entry.path, paths)
    return paths

def extract_files_from_tar(file_path, target_path):
    # Check if the target_path directory exists
    extracted_files=[]
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with tarfile.open(file_path) as tar:
        for member in tar.getmembers():
            if member.isfile():  # check if it is a file
                # To extract only files, make sure the output filename does not include any directory structure
                filename = os.path.basename(member.name)
                member.name = filename  # reset the member name to just the filename
                tar.extract(member, path=target_path)  # extract the file
                extracted_files.append(target_path+"/"+member.name)
    return extracted_files


def download_file(url, destination):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Check if the request header contains the file size information
        file_size = response.headers.get('Content-Length')
        if file_size is not None:
            file_size = int(file_size)

        downloaded_size = 0
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=2**21):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    # If file size is known, we can calculate the progress
                    if file_size is not None:
                        print(f"Downloaded: {downloaded_size / 1024 / 1024:.2f}MB of {file_size / 1024 / 1024:.2f}MB", end='\r')
                    else:
                        print(f"Downloaded: {downloaded_size / 1024 / 1024:.2f}MB", end='\r')

        print(f"\nDownload finished. The file was saved to {destination}.")
    else:
        print(f"Failed to download the file. Server responded with status code {response.status_code}.")

def upload_file_s3(filepath, key):
    # validate the file path
    if not os.path.isfile(filepath):
        raise ValueError("The file does not exist or is not a file: {}".format(filepath))

    # validate and retrieve environment variables
    if "AWS_REGION" not in os.environ or "AWS_ACCESS_KEY_ID" not in os.environ or "AWS_SECRET_ACCESS_KEY" not in os.environ or "BUCKET_NAME" not in os.environ:
        raise EnvironmentError("Required environment variables are not set")

    region = os.getenv('AWS_REGION')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('BUCKET_NAME')

    try:
        # Creating a resource for 's3' 
        s3_resource = boto3.resource(
            's3', 
            region_name = region, 
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key
        ) 

        # Upload a file to S3 bucket
        s3_resource.Bucket(bucket_name).put_object(
            Key = key, 
            Body = open(filepath, 'rb')
        )

        # Creating a client for 's3' 
        s3_client = boto3.client(
            's3', 
            region_name = region, 
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key
        )
        try:
            # Check if the file was uploaded successfully
            s3_client.head_object(Bucket=bucket_name, Key=key)
            print("File was uploaded successfully")
            # Generate the URL to get 'key-name' from 'bucket-name'
            url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': key
                }
            )
            return url

        except ClientError:
            # The file wasn't found. 
            return "File was not found in the bucket. Upload failed."
    except NoCredentialsError:
        return "No AWS credentials were found"

def generate_random_string(length=5):
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(length))

if __name__ == "__main__":
    outFile = "report-"+generate_random_string()+".pdf" #just in case the tar file has no name
    parser = argparse.ArgumentParser(description="""FTDC Decoder Script. 
    This script accepts either a tar/directory of input files or a URL to fetch the input data. 
    It also requires a timestamp to process the data, and an output file path where the results will be written.
    The interval parameter determines the data aggregation period in minutes. An 'exact' parameter can be set to 1, if we do not want to search for a ticket and assume there is a drop ticket present.""")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=validate_directory, help="Input directory path")
    input_group.add_argument("--inputUrl", type=validate_url, help="Input URL")

    parser.add_argument("--timestamp", type=validate_timestamp, required=True, help="Milliseconds from epoch")
    parser.add_argument("--output", type=str, required=False,default=outFile, help="Output file path")
    parser.add_argument("--interval", type=validate_interval, required=False, default=5, help="Interval in minutes")
    parser.add_argument("--exact", type=int, required=False, default=0, help="set the timestamp as drop time t0")
    args = parser.parse_args()
    print("Requested Query Timestamp:",args.timestamp)
    destination=""
    tarfilename=""
    if args.inputUrl:
        file_url = args.inputUrl
        tarfilename=file_url[file_url.rindex('/')+1:file_url.index('.tar')]
        file_name = generate_random_string(3)+"_"+tarfilename
        download_file(file_url,file_name)
        destination=file_name
    elif args.input and os.path.isfile(args.input):
        destination=args.input
        tarfilename=destination[destination.rindex('/')+1:destination.index('.tar')]
    if destination != "":
        out_folder = generate_random_string(10)
        os.mkdir(out_folder)
        files=extract_files_from_tar(destination,out_folder)
    else:
        files=find_files(args.input)
        
    st=time.time()
    KEY_NAME= 'OPENAI_API_KEY'
    key = os.environ.get(KEY_NAME)
    if key is None:
        print("Please ensure there is an API KEY in the environment variables under OPENAI_API_KEY")
        exit(1)
    # files = dirPath.glob("*")
    filtered_files=[]
    all_files= [i for i in files if "metrics" in i]
    all_files.sort()
    # print(all_files)
    for file in all_files:
        file_name=file[file.rindex('/')+1:]
        if "interim" not in file_name:
            tstamp=convert_to_datetime(file_name[file_name.index('.')+1:])
            diff=abs(tstamp-args.timestamp)
            if diff <= timedelta(hours=6):
                filtered_files.append(file)
    filtered_files.sort()
    if len(filtered_files)==0:
        raise ValueError("No files corresponding to the queryTimestamp found. Please check the timestamp/path and try again!")
    if all_files.index(filtered_files[-1]) == len(all_files)-2: # metrics.interim should be included
        filtered_files.append(all_files[-1])
    report_filename = "report-"+tarfilename+".pdf"
    decoder = FTDC(filtered_files,args.timestamp,report_filename,args.interval*60, args.exact)
    decoder.process()
    # st=time.time()-st
    if os.path.isfile(report_filename):
        print("report locally saved as:",report_filename)
        downloadUrl = upload_file_s3(report_filename,report_filename)
        print(downloadUrl)
    if destination!="":
        shutil.rmtree(out_folder)
    # print("Runtime in seconds: ",st)