# Run the SageWorks Dashboard

You can run the Dashboard on a local environment:

```bash 
$ cd applications/aws_dashboard
$ pip install -r requirements.txt
$ python app.py
``` 

Or you can run it in a Docker container:

```bash
$ cd applications/aws_dashboard
$ docker build -t aws_dashboard .
$ docker run -p 8050:8050 aws_dashboard
```