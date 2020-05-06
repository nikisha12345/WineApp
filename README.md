# Notes 
1. Training, validation and trained model data were uploaded to S3 Bucket and it's url : `s3://wineappcloud/` 

# To Run Model Training on 4 Parallel EC2 Instances using EMR on AWS
1. Login to AWS Console and Create a IAM role for ec2 instance to give access to s3 so that ec2 instace can have access to download (CSV file) and upload files (Model file) to s3
2. Create cluster using following steps
	- In `Launch mode`, select `Step execution`
	- In `Step type`, select `Spark application` & click on `Configure`, set `Deploy mode` = `Client`, set `Spark-submit options` = `--packages org.apache.hadoop:hadoop-aws:2.7.7`, `Application location` = `s3://wineappcloud/WineApp.py` and set `Action on failure` = `Terminate Cluster` (It will terminate after successfull completion of the step)
	- In `Software configuration`, choose `spark`
	- In `Hardware configuration` , choose `Number of instances` 4
	- Hit create Cluster, It will run for around 30 minutes. 
	- You can see the standard output via clicking `Step` tab click on View logs for `Spark application`
	OR you can create this cluster in one line using below command using aws cli

```bash
aws emr create-cluster --applications Name=Hadoop Name=Spark --ec2-attributes '{"InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-5a042c64","EmrManagedSlaveSecurityGroup":"sg-0233a1837a16a7902","EmrManagedMasterSecurityGroup":"sg-0a31cc8dff123ed0b"}' --release-label emr-5.29.0 --log-uri 's3n://aws-logs-700559207820-us-east-1/elasticmapreduce/' --steps '[{"Args":["spark-submit","--deploy-mode","client","--packages","org.apache.hadoop:hadoop-aws:2.7.7","s3://wineappcloud/WineApp.py"],"Type":"CUSTOM_JAR","ActionOnFailure":"TERMINATE_CLUSTER","Jar":"command-runner.jar","Properties":"","Name":"Spark application"}]' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master Instance Group"},{"InstanceCount":3,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"CORE","InstanceType":"m5.xlarge","Name":"Core Instance Group"}]' --configurations '[{"Classification":"spark","Properties":{}}]' --auto-terminate --service-role EMR_DefaultRole --enable-debugging --name 'WineApp' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-east-1
```


# To Run Model Prediction on single EC2 Instance without Docker
### Assumptions 
- Ec2 instance is created
- It has spark installed on it
1. Copy the wine_prediction_final.py (https://github.com/nikisha12345/WineApp/blob/master/wine_prediction_final.py) file to the Ec2 instance (scp -i <"your .pem file"> wine_prediction_final.py <Ec2 instance ID>:~/wine_prediction_final.py)
2. Run the following command in Ec2 instance to start the model prediction : 
- Example of using S3 file as argument 
`spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 wine_prediction_final.py --test_file s3a://wineappcloud/ValidationDataset.csv` (Here you can pass S3 link of the test.data just like this)
- Example of using local file as argument
`spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 wine_prediction_final.py --test_file ValidationDataset.csv` (Here ValidationDataset.csv should be in your Ec2 instance)

# To Run Model Prediction on single EC2 Instance with Docker
### Assumptions
- Ec2 instance is created
- It has docker installed on it
1. `ssh into your Ec2 instance` 
2. `docker pull nikisha25/wineapp`
3. `docker run -it nikisha25/wineapp:latest --test_file s3a://wineappcloud/ValidationDataset.csv ( Here you can pass the S3 link)`


