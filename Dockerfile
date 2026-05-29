FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your local code into the container
COPY . .

CMD ["python", "train.py"]