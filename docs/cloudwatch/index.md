# SageWorks CloudWatch

!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and SageWorks. So please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

The SageWorks framework continues to flex to support different real world use cases when operating a set of production machine learning pipelines. As part of this we're including CloudWatch log forwarding/aggregation for any service using the SageWorks API (Dashboard, Glue, Lambda, Notebook, Laptop, etc).


### Log Groups and Streams
The SageWorks logging setup includes the addition of a CloudWatch 'Handler' that forwards all log messages to the `SageWorksLogGroup`


**Individual Streams**

Each process running SageWorks will get a unique individual stream.

- **ecs/DashBoard\*** (any logs from SageWorks Dashboard)
- **glue/\*** (logs from Glue jobs)
- **lambda/\*** (logs from Lambda jobs)
- **docker/\*** (logs from Docker containers)
- **laptop/\*** (logs from laptop/notebooks)

Since many jobs are run nightly/often, the stream will also have a date on the end... `glue/my_job/2024_08_01_17_15`

### AWS CloudWatch made Easy
!!!tip inline end "Logs in Easy Mode"
    The SageWorks `cloud_watch` command line tool gives you access to important logs without the hassle. Automatic display of important event and the context around those events.

```
pip install sageworks
cloud_watch
```

The `cloud_watch` script will automatically show the interesting (WARNING and CRITICAL) messages from any source within the last hour. There are lots of options to the script, just use `--help` to see options and descriptions.

```
cloud_watch --help
```

Here are some example options:

```
# Show important logs in last 12 hours
cloud_watch --start-time 720 

# Show a particular stream
cloud_watch --stream glue/my_job 

# Show/search for a message substring
cloud_watch --search SHAP

# Show a log levels (matching and above)
cloud_watch --log-level WARNING
cloud_watch --log-level ERROR
cloud_watch --log-level CRITICAL
OR
cloud_watch --log-level ALL  (for all events)

# Combine flags 
cloud_watch --log-level ERROR --search SHAP
cloud_watch --log-level ERROR --stream Dashboard
```

These options can be used in combination and try out the other options to make the perfect log search :)

<img alt="sageworks cloud_watch" src="https://github.com/user-attachments/assets/820817de-8f32-47e8-98dc-f3d3f415b2ea">

### More Information
Check out our presentation on [SageWorks CloudWatch](https://docs.google.com/presentation/d/1Jtoo7LXWBSF2xCpn9BNLQlnAtN2vIELCzn_-XMu9GAI/edit?usp=sharing)

### Access through AWS Console
Since we're leveraging AWS functionality you can always use the AWS console to look/investigate the logs. In the AWS console go to **CloudWatch... Log Groups... SageWorksLogGroup**

<img alt="sageworks log group" src="https://github.com/user-attachments/assets/a7778232-08db-4950-952c-dd8de650bae8">

    
## Questions?
<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and SageWorks. Please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

