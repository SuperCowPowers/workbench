# Use the official Amazon Linux 2 image for AWS Lambda
FROM amazon/aws-lambda-python:3.10 as build

# Create directory for dependencies and set it as the working directory
WORKDIR /asset-output/python

# Install SageWorks (includes dependencies) into target directory
RUN pip install --no-cache-dir sageworks==0.6.16 -t .

# Strip binary files to reduce size
RUN find . -name "*.so" -exec strip --strip-unneeded {} \;

# Remove unnecessary files (tests, docs, examples)
RUN find . -type d -name "tests" -exec rm -rf {} + \
    && find . -type d -name "docs" -exec rm -rf {} + \
    && find . -type d -name "examples" -exec rm -rf {} +

# Create the final image
FROM amazon/aws-lambda-python:3.10

# Copy dependencies from the build stage
COPY --from=build /asset-output /asset-output

# Set working directory
WORKDIR /asset-output

# Override the entrypoint to start a shell
ENTRYPOINT ["/bin/bash"]

# List contents to verify installation and size breakdown
RUN du -sh /asset-output/* /asset-output/python/* | sort -h
