# Use Python 3.13 slim base image
FROM python:3.13-slim

# Set working directory in the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]