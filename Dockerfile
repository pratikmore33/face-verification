FROM python:3.9

# Install system-level dependencies necessary for dlib compilation
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /code

# Copy requirements.txt to the working directory 
COPY requirements.txt /code/requirements.txt

# Copy all files to the working directory
COPY . /code/

# Install requirements including dlib
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Run FastAPI server with the specified command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
