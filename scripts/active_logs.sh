# Get all log group names
log_groups=$(aws logs describe-log-groups --query 'logGroups[*].logGroupName' --output text)

# Set the start and end time for the last hour
start_time=$(($(date +%s) - 3600))000
end_time=$(($(date +%s)))000

# Initialize an empty list for updated log groups
updated_log_groups=()

# Loop through each log group and check for updates
for log_group in $log_groups; do
    # Check if there are any log events in the last hour
    events=$(aws logs filter-log-events \
        --log-group-name "$log_group" \
        --start-time "$start_time" \
        --end-time "$end_time" \
        --query 'events' \
        --output json)

    # If events are found, add the log group to the updated list
    if [ "$events" != "[]" ]; then
        updated_log_groups+=("$log_group")
    fi
done

# Output the updated log groups
if [ ${#updated_log_groups[@]} -eq 0 ]; then
    echo "No log groups have been updated in the last hour."
else
    echo "Log groups updated in the last hour:"
    for log_group in "${updated_log_groups[@]}"; do
        echo "$log_group"
    done
fi
