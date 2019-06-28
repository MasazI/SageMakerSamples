import boto3
from sagemaker import get_execution_role
from sagemaker.local import LocalSession

# AWS profile for this session
boto_sess = boto3.Session(profile_name='<profile name>', region_name='<region name>')

# sage maker local session
local_session = LocalSession(boto_sess)

# S3 bucket name
bucket = '<bucket name>'

# upload path for endpoint script.
custom_code_upload_location = 's3://{}/customcode/tensorflow_iris'.format(bucket)

# the path saved artifacts (outputs of training)
model_artifacts_location = 's3://{}/artifacts'.format(bucket)

# IAM Role
# For only local mode, it's not used but need to exits.
role = '<role name>'

from sagemaker.tensorflow import TensorFlow

# TensorFlow Estimator
iris_estimator = TensorFlow(entry_point='./iris_dnn_classifier.py',
                            role=role,
                            framework_version='1.12.0',
                            output_path=model_artifacts_location,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            train_instance_type='local',
                            training_steps=20,
                            evaluation_steps=1,
                            sagemaker_session=local_session)

# Data path
region = boto_sess.region_name
train_data_location = 's3://sagemaker-sample-data-{}/tensorflow/iris'.format(region)

# Start training
iris_estimator.fit(train_data_location)