# CloudWatch

!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and SageWorks. So please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

The SageWorks framework continues to 'flex' to support different real world use cases when operating a set of production machine learning pipelines. As part of this we're including CloudWatch log forwarding for any use of the SageWorks API (Dashboard, Glue, Lambda, Notebook, Laptop, etc).


### Functionality
The SageWorks logging setup includes the addition of a CloudWatch 'Handler' that forwards all log messages to the `SageWorksLogGroup`

<img alt="private_saas_compare" src="https://github.com/user-attachments/assets/a7778232-08db-4950-952c-dd8de650bae8">

### Individual Streams
Each process running SageWorks will get a unique individual stream.

- **dashboard/\*** (any logs from Web Dashboard)
- **glue/\*** (logs from Glue jobs)
- **lambda/\*** (logs from Lambda jobs)
- **docker/\*** (logs from Docker containers)
- **laptop/\*** (logs from laptop/notebooks)

Since many jobs are run nightly/often, the stream will also have a date on the end... `glue/my_job/2024_08_01_17_15`
    
### Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to anser any questions you may have about AWS and SageWorks. Please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 


