FROM python:3.10.11

# Install dependencies
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]