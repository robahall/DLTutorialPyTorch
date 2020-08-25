import subprocess
import logging
from pathlib import Path, PurePath
from io import BytesIO
import datetime

from src import ROOT_DIR
from tqdm import tqdm, trange
import requests
from bs4 import BeautifulSoup
import zipfile
import boto3
from botocore.exceptions import ClientError


def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def find_files_website(web_address):
    """
    Find applicable files on website.
    :param web_address: (str) web address
    :return: (list) links with test
    """
    root_web = web_address.split('/r')[0]
    soup = BeautifulSoup(requests.get(web_address).text, features="lxml")
    links_with_text = []
    for a in soup.find_all('a', href=True):
        if a.text and a['href'].endswith('download=1'):
            links_with_text.append(f'{root_web}{a["href"].rstrip("?download=1")}')
    logging.info(set(links_with_text))
    return set(links_with_text)

def extract_file_name(file_address):
    return str(file_address).split("/")[-1]

def file_path(file_name):
    return Path("/home/rob/Documents/AWS_DL") / file_name

def extract_parent_name(file_address):
    """
    Extract file names and suffix
    :param file_address: (str) full filename
    :return: (str) filename, suffix
    """
    file_split = str(file_address).split("/")[-1].split(".")
    return file_split[0], file_split[1]

def download_luna_file(file_address):
    """
    Run shell script to download filename from website to dir.
    :param filename: (str) file name to download
    :return:
    """
    try:
        subprocess.run(['./download.sh'], input=str(file_address), text=True)
        logging.info(f"{file_address} successfully downloaded.")
        return True
    except Exception as e:
        logging.error(e)
        return False

def delete_luna_file(filename):
    """
    Delete file after upload and unzip.
    :param filename: (str) filename to delete
    :return:
    """
    try:
        subprocess.run(['./remove.sh'], input=str(filename), text=True)
        logging.info(f"{filename} successfully deleted.")
    except Exception as e:
        logging.error(e)
        return False
    return True


def upload_file_s3(filename, bucket, object_name=None):
    """
    Upload a file to an S3 bucket
    :param filename: (Path or Path-like str)
    :param bucket: (str) Bucket to upload to
    :param object_name: (str) S3 object name. If not specified then file_name is used
    :return:
    """
    if object_name is None:
        object_name = PurePath(filename).name


    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(str(filename), bucket, f'luna/{object_name}')
        logging.info(f"{response}")
    except ClientError as e:
        logging.error(e)
        return False
    return True


def get_bucket_names():
    """
    Get current bucket names
    :return:
    """
    s3 = boto3.client('s3')
    response = s3.list_buckets()

    print('Existing buckets:')
    for bucket in response['Buckets']:
        print(f' {bucket["Name"]}')

def get_s3_keys(prefix):
    """
    Return keys in buckets.
    :param prefix: (str)
    :return:
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('robml')
    for obj in bucket.objects.filter(Prefix=prefix):
        print(obj.key)


def unzip_file_s3(filename, parent_name):
    s3_resource = boto3.resource('s3')
    zip_obj = s3_resource.Object(bucket_name='robml', key=f'luna/{filename}')
    buffer = BytesIO(zip_obj.get()["Body"].read())

    z = zipfile.ZipFile(buffer)
    for fname in tqdm(z.namelist(), total=len(z.namelist()), unit="files"):
        try:
            file_info = z.getinfo(fname)
            s3_resource.meta.client.upload_fileobj(z.open(fname),
                                                Bucket='robml',
                                                Key=f'luna/{parent_name}/{fname}'
                                                )
            logging.info(f"{fname} successfully extracted.")
        except ClientError as e:
            logging.error(e)

def main():
    log_path = ROOT_DIR / "logs"
    logging.basicConfig(filename=f"{log_path / current_time()}.log", level=logging.INFO)
    web_address = ["https://zenodo.org/record/3723295", "https://zenodo.org/record/3723299"]
    files = []
    for address in web_address:
        files.append(find_files_website(address))
    files = [address for sublist in files for address in sublist] # list of files from website

    for file in tqdm(files, total=len(files), unit="files"):
        parent, file_type = extract_parent_name(file)
        fname = extract_file_name(file)
        #download
        download_luna_file(file)
        #upload
        current_file_loc = file_path(fname)
        upload_file_s3(current_file_loc, bucket="robml")
        #unzip
        if file_type == "zip":
            unzip_file_s3(fname, parent)
        #delete
        delete_luna_file(current_file_loc)



if __name__ == "__main__":
    main()



