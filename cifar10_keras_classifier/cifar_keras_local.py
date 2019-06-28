import boto3
from sagemaker import get_execution_role
from sagemaker.local import LocalSession

# AWS profile for this session
boto_sess = boto3.Session(profile_name='<profile_name>', region_name='<region_name>')

# sage maker local session
local_session = LocalSession(boto_sess)

# S3 bucket name
bucket = '<bucket_name>'

# upload path for endpoint script.
custom_code_upload_location = 's3://{}/customcode/tensorflow_cifar'.format(bucket)

# the path saved artifacts (outputs of training)
model_artifacts_location = 's3://{}/artifacts_cifar'.format(bucket)

# IAM Role
# For only local mode, it's not used but need to exits.
role = '<role name>'

from sagemaker.tensorflow import TensorFlow

# TensorFlow Estimator
estimator = TensorFlow(entry_point='cifar10_cnn.py',
                       role=role,
                       framework_version='1.12.0',
                       hyperparameters={'learning_rate': 1e-4, 'decay':1e-6},
                       training_steps=1000, evaluation_steps=100,
                       output_path=model_artifacts_location,
                       code_location=custom_code_upload_location,
                       train_instance_count=1,
                       train_instance_type='local',  # local model
                       sagemaker_session=local_session)

# Data path
inputs = local_session.upload_data(path='./data/cifar10', key_prefix='data/DEMO-cifar10')

# Start training
estimator.fit(inputs)