# Deploying S3 Plugins with the Dashboard

!!!tip inline end "Need Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and SageWorks. So please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

Notes and information on how to include S3 based plugins with your SageWorks Dashboard. 

Deploying your Dashboard plugins via an S3 bucket allows plugin developers to modify and improve plugins without a bunch of Docker builds, ECR, and CDK Deploy.

On, the other hand when you have a 'final build' you might want to lock it in with a Docker image. If you'd like to bundle up your plugins in a Docker image and deploy please see [Dashboard with Plugins](dashboard_with_plugins.md).

## Check your Dashboard
Make sure your Dashboard is configured to pull plugin pages, views, and web_components from an S3 bucket. Go to your main dashboard page and there's a 'secret link' when you click on the main title that brings up the **Dashboard Status Page**.

<img src="../images/dashboard_secret_click.png" width="500">

**Dashboard Status Page**
Okay now check your **Plugin Path:** config and make sure it points to the S3 bucket you're expecting.

<img src="../images/status_showing_S3_path.png" width="350">

## Develop your Plugins
During development it's good to both unit testing and local dashboard testing. Please see our main plugin docs for how to do local testing [Plugins General](../plugins/index.md).

When you're ready to 'deploy' the plugins you can copy them up to the S3 bucket/prefix. You want to copy recursively so if you're plugin directory looks like the listing below you want to copy all the files/directories, so that the dashboard picks up everything.

```
- my_plugins
   - pages
      - page_1.py
   - views
      - view_1.py
   - web_components
       -component_1.py
       -component_2.py
```

!!! warning "Careful with Plugin Copy" 
    In particular, pay attention to additional files in the directory structure. You do not want to copy \_pycache\_ and \*.pyc files. So we recommend using this CLI.

```
cd my_plugins
aws s3 cp . s3://my_bucket/prefix --recursive --exclude "*" --include "*.py"
```


## Restart the ECS Service
Okay, so this is a bit heavy handed, but automatically removing/adding/modifying the plugin pages, views, and web_components was 'amazingly complicated' and will be a feature request for a later date. :)

**Getting Cluster and Service Names**

You can go to the AWS Console, Elastic Container Service, find the cluster, click on that and find the service.

The cluster will be something like:

```
SageworksDashboard-SageworksCluster123456
```

and the service will be something like:

```
SageworksDashboard-SageworksService789123
```

Anyway, find those two things and run this command below (**Note:** You probably need admin permisions)

```
aws ecs update-service --cluster your-cluster-name \
--service your-service-name --force-new-deployment
```

**Important:** Even though this command will finish immediately, the ECS service will slowly flip over to the new instance (like 5-10 minutes), so wait a bit before testing the changes.

### Verify new Plugin changes
Okay now that the ECS service has restarted (which can take a bit) you can now go to the Dashboard and test/verify that the changes you made now show up on the Dashboard.

```
docker build -t my_sageworks_with_plugins:v1_0 --platform linux/amd64 .
```

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and SageWorks. Please contact us at [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 