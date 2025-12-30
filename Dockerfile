FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for user
WORKDIR $HOME/app

# Copy requirements first (for better caching)
COPY --chown=user requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user src/ ./src/
COPY --chown=user app.py ./

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV HOST=0.0.0.0

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
