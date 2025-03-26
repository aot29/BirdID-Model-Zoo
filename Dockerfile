# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install the Docker CLI and any other OS dependencies
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/ouptput
RUN mkdir -p /app/data


# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files (update paths as necessary)
COPY runModel.py /app/runModel.py
COPY LabelToIdMappings/ /app/LabelToIdMappings/
RUN touch /app/inputPaths.txt


# Set the entrypoint to run the model script
ENTRYPOINT ["python", "runModel.py"]
