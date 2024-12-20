# AWS Credentials Lock

!!!tip inline end "AWS Curious?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

## TBD


## Threading Lock
A threading lock object in Python is used to ensure that only one thread can access a specific section of code or resource at a time, preventing race conditions.

**botocore/credentials.py**
```
self._refresh_lock = threading.Lock()
```



## Error
```
elif self.refresh_needed(self._mandatory_refresh_timeout):
    # If we're within the mandatory refresh window,
    # we must block until we get refreshed credentials.
    with self._refresh_lock:
        if not self.refresh_needed(self._mandatory_refresh_timeout):
            return
        self._protected_refresh(is_mandatory=True)
```



## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 



