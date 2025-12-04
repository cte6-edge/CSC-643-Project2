# AWS-Project
CS643 – Project 2 (Training + Prediction Application)
Author: Clifford Edghill
Date: 12/04/2025

This document describes the end-to-end setup for CS643 Project 2, including:
1) Cloud environment configuration  
2) Parallel Spark model training on 4 EC2 instances  
3) Running the prediction application WITHOUT Docker  
4) Running the prediction application WITH Docker  

All resources are created in the N. Virginia region (us-east-1).

Region: us-east-1

Training Dataset: s3://<YOUR_BUCKET>/TrainingDataset.csv  
Validation Dataset: s3://<YOUR_BUCKET>/ValidationDataset.csv  
Prediction Test File: ValidationDataset.csv (renamed to TestDataset.csv)  
Model Output Folder: s3://<YOUR_BUCKET>/wine_model/

NOTE: IAM roles are NOT used for this project.  
AWS Access Keys must be configured *manually* on every EC2 instance.

------------------------------------------------------------
--AWS Access Key Setup (NO IAM ROLES)
------------------------------------------------------------
1. In the AWS Console → go to “My Security Credentials”.
2. Create an **Access Key** (Access Key ID + Secret Access Key).
3. On every EC2 instance, configure:

   aws configure
   → Enter Access Key ID  
   → Enter Secret Access Key  
   → Region: us-east-1  
   → Output: json  

4. If a session token is provided by your class or Learner Lab:

   aws configure set aws_session_token "PASTE_SESSION_TOKEN_HERE"

------------------------------------------------------------
--S3 Setup
------------------------------------------------------------
1. Open AWS → S3 → “Create bucket”.
2. Bucket name: <your-unique-bucket>
3. Region: us-east-1
4. Upload the required datasets:
   - TrainingDataset.csv
   - ValidationDataset.csv
5. Create a folder for model output:
   wine_model/

------------------------------------------------------------
--EC2 Setup (4 Instances for Spark Training)
------------------------------------------------------------
Create 4 instances:

● 1 Spark Master  
● 3 Spark Workers  

All instances use:

● AMI: Ubuntu Server 24.04  
● Instance Type: t3.medium  
● Key Pair: None (use EC2 Instance Connect)  

Network → Security Group “cs643-sg”
Inbound Rules:
- SSH (22) 0.0.0.0/0  
- TCP 8080 (Spark UI)  
- TCP 7077 (Spark Master)  
- TCP 4040 (Spark Jobs UI)  
- All traffic from same security group (cs643-sg)

Enable Auto-assign Public IP.

Launch:
spark-master  
spark-worker-1  
spark-worker-2  
spark-worker-3  

Wait for “2/2 checks passed”.

------------------------------------------------------------
--Connect to Instances (EC2 Instance Connect)
------------------------------------------------------------
For each instance:
Select → Connect → EC2 Instance Connect → Connect

------------------------------------------------------------
--Install Java, Spark, Python on ALL 4 Instances
------------------------------------------------------------
sudo apt update -y
sudo apt install openjdk-17-jdk python3-pip -y

wget https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar -xvf spark-3.5.1-bin-hadoop3.tgz
sudo mv spark-3.5.1-bin-hadoop3 /opt/spark

Add to ~/.bashrc:

export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

Reload:
source ~/.bashrc

------------------------------------------------------------
--Spark Cluster Configuration
------------------------------------------------------------

### On ALL workers:
Edit /opt/spark/conf/workers:

spark-worker-1  
spark-worker-2  
spark-worker-3  

### On ALL nodes:
Add private IPs:

sudo bash -c 'cat >> /etc/hosts' <<EOF
<MASTER_PRIVATE_IP> spark-master
<WORKER1_PRIVATE_IP> spark-worker-1
<WORKER2_PRIVATE_IP> spark-worker-2
<WORKER3_PRIVATE_IP> spark-worker-3
EOF

### On Master:
start-master.sh  
start-worker.sh spark://spark-master:7077  
start-worker.sh spark://spark-master:7077  
start-worker.sh spark://spark-master:7077  

Check Spark UI:
http://<MASTER_PUBLIC_IP>:8080

------------------------------------------------------------
--Download and Place Training Script
------------------------------------------------------------
Place train.py into:

~/cs643/train/train.py

------------------------------------------------------------
--Run Distributed Training Job
------------------------------------------------------------
spark-submit \
  --master spark://spark-master:7077 \
  train.py \
  --train s3://<YOUR_BUCKET>/TrainingDataset.csv \
  --valid s3://<YOUR_BUCKET>/ValidationDataset.csv \
  --output s3://<YOUR_BUCKET>/wine_model/

Output:
✓ Best F1 score printed  
✓ Model saved (wine_model.zip) in S3 bucket  
✓ Training completed across all 4 EC2 instances  

------------------------------------------------------------
--Prediction Application (RUN ON SINGLE EC2 INSTANCE)
------------------------------------------------------------
Launch one EC2 instance:

● Name: prediction-instance  
● AMI: Ubuntu 24.04  
● Instance Type: t3.small or t3.medium  
● Security group: allow SSH only  

Connect using EC2 Instance Connect.

------------------------------------------------------------
--Download Model & Dataset for Testing
------------------------------------------------------------
mkdir ~/cs643/predict
cd ~/cs643/predict

aws s3 cp s3://<BUCKET>/wine_model/wine_model.zip .
aws s3 cp s3://<BUCKET>/ValidationDataset.csv ./TestDataset.csv

------------------------------------------------------------
--Run Prediction WITHOUT Docker
------------------------------------------------------------
sudo apt update -y
sudo apt install openjdk-17-jdk python3-pip -y

wget https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar -xvf spark-3.5.1-bin-hadoop3.tgz
sudo mv spark-3.5.1-bin-hadoop3 /opt/spark

pip3 install pyspark pandas

Run:

python3 predict.py TestDataset.csv

Outputs:
✓ Cleaned dataset  
✓ Model loaded  
✓ Predictions printed  
✓ F1 Score displayed  

------------------------------------------------------------
--Run Prediction WITH Docker
------------------------------------------------------------

### 1. Install Docker
sudo apt update  
sudo apt install docker.io -y  
sudo systemctl enable --now docker  

### 2. Build Docker Image
Inside ~/cs643/predict place:

● predict.py  
● wine_model.zip  
● TestDataset.csv  
● Dockerfile  

Build:

sudo docker build -t cs643-wine-predict .

### 3. Run Container

sudo docker run --rm -it \
  -v "$(pwd)/TestDataset.csv:/app/TestDataset.csv" \
  -v "$(pwd)/wine_model.zip:/app/wine_model.zip" \
  cs643-wine-predict \
  /app/TestDataset.csv

Outputs:
✓ Model extracted  
✓ Spark local session started  
✓ Predictions generated  
✓ Works without any Spark cluster  

------------------------------------------------------------
--Docker Hub Submission Link
------------------------------------------------------------
Push the built image to Docker Hub:

sudo docker tag cs643-wine-predict cte6njit/csc-643-project2:latest  
sudo docker push cte6njit/csc-643-project2:latest  

Your public Docker Hub link:
https://hub.docker.com/r/cte6njit/csc-643-project2

------------------------------------------------------------
--End of Project README
------------------------------------------------------------
