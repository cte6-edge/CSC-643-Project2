# Use a small Python base image
FROM python:3.11-slim

# Install Java (needed for Spark)
RUN apt-get update && \
    apt-get install -y openjdk-21-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Working directory inside the container
WORKDIR /app

# Copy the prediction script into the image
COPY predict.py /app/predict.py

# Install Python dependencies (no cache to keep image smaller)
RUN pip install --no-cache-dir pyspark pandas numpy

# Default entrypoint:
#   docker run ... cs643-image /app/TestDataset.csv
ENTRYPOINT ["python", "predict.py"]
