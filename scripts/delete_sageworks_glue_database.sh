# Delete all tables in the Glue database
for table in $(aws glue get-tables --database-name sageworks --query 'TableList[].Name' --output text); do
    echo "Deleting table: $table"
    aws glue delete-table --database-name sageworks --name "$table"
done

# Delete the Glue database
echo "Deleting database: sageworks"
aws glue delete-database --name sageworks
