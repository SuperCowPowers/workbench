# Run the Anomaly Inspector

You can run the Anomaly Inspector in a local environment:

```bash 
$ cd applications/anomaly_inspector
$ pip install -r requirements.txt
$ python app.py
``` 

Or you can run it in a Docker container:

```bash
$ cd applications/anomaly_inspector
$ docker build -t anomaly_inspector .
$ docker run -p 8050:8050 anomaly_inspector
```