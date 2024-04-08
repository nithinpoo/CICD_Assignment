# Use an appropriate base image, such as Python.
FROM python:3.9

# Set the working directory inside the container.
WORKDIR /app

# Copy the entire repository contents into the container at /app.
COPY . .

# Install dependencies listed in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Command to train the model during the image build phase.
RUN python train.py

# Command to execute test script when the container is run.
CMD ["python", "test.py"]
