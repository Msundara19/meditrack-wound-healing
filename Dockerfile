# Use Linux base image so Pathway works
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Disable Streamlit usage stats prompts
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Default command: run the dashboard
CMD ["streamlit", "run", "src/meditrack/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
