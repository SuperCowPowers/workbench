from workbench.api.monitor import Monitor

# Create an Endpoint Monitor Class and perform initial Setup
endpoint_name = "abalone-regression-end-rt"
mon = Monitor(endpoint_name)

# Add data capture to the endpoint
mon.add_data_capture(capture_percentage=100)

# Create a baseline for monitoring
mon.create_baseline()

# Set up the monitoring schedule
mon.create_monitoring_schedule(schedule="hourly")
