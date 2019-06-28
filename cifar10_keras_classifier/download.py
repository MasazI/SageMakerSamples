import os
import boto3
import tarfile

data_dir = './data/cifar10'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if os.path.exists(os.path.join(data_dir, 'cifar-10-batches-bin')):
    print('cifar dataset already downloaded')
    exit()

filename = 'cifar-10-binary.tar.gz'
filepath = os.path.join(data_dir, filename)

if not os.path.exists(filepath):
    boto_sess = boto3.Session(profile_name='<profile name>', region_name='<region name>')
    boto_sess.resource('s3', region_name='<region name>').Bucket('sagemaker-sample-data-{}'.format('<region name>')).download_file(
        'tensorflow/cifar10/cifar-10-binary.tar.gz', './data/cifar10/cifar-10-binary.tar.gz')
tarfile.open(filepath, 'r:gz').extractall(data_dir)
