import subprocess
import requests
import re
import logging
from bs4 import BeautifulSoup
import zipfile
import boto3
from botocore.exceptions import ClientError
from io import BytesIO

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
    return set(links_with_text)

def extract_file_name(file_address):
    return file_address.split("/")[-1]

def download_luna_file(file_address):
    """
    Run shell script to download filename from website to dir.
    :param filename: (str) file name to download
    :return:
    """
    try:
        subprocess.run(['./download.sh'], input=file_address)
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
        subprocess.run(['./remove.sh'], input=filename)
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
        object_name = extract_file_name(filename)


    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(filename, bucket, f'luna/{object_name}')
    except ClientError as e:
        logging.error(e)
        return False
    return True


def get_bucket_names():
    s3 = boto3.client('s3')
    response = s3.list_buckets()

    print('Existing buckets:')
    for bucket in response['Buckets']:
        print(f' {bucket["Name"]}')

def get_s3_keys(prefix):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('robml')
    for obj in bucket.objects.filter(Prefix=prefix):
        print(obj.key)


def unzip_file_s3(filename):
    s3_resource = boto3.resource('s3')
    zip_obj = s3_resource.Object(bucket_name='robml', key=f'luna/{filename}')
    buffer = BytesIO(zip_obj.get()["Body"].read())

    z = zipfile.ZipFile(buffer)
    for fname in z.namelist():
        try:
            file_info = z.getinfo(fname)
            s3_resource.meta.client.upload_fileobj(z.open(fname),
                                                Bucket='robml',
                                                Key=f'luna/candidates_v2/{fname}'
                                                )
        except ClientError as e:
            print(e)

def main():
    web_address = ["https://zenodo.org/record/3723295", "https://zenodo.org/record/3723299"]
    files = []
    for address in web_address:
        files.append(find_files_website(address))
    files = [address for sublist in files for address in sublist] # list of files from website

    for file in files:
        #download
        download_luna_file(file)

        #upload
        #unzip
        #delete



    return list(files)


if __name__ == "__main__":
    #unzip_file_s3("candidates_v2.zip")
    #extract_file_name("https://zenodo.org/record/3723295/files/annotations.csv")
    #print(find_files_website("https://zenodo.org/record/3723295"))
    #download_luna_file(b"https://zenodo.org/record/3723295/files/annotations.csv")
    from pathlib import Path
    upload_file_s3("/home/rob/Documents/AWS_DL/annotations.csv", bucket="robml")
    ## TODO check if file ends in zip. If it does then extract. Else continue.



