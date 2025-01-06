# AWS Domain and Certificate Instructions
!!!tip inline end "Need AWS Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

This page tries to give helpful guidance when setting up a new domain and SSL Certificate in your AWS Account.

## New Domain
You'll want the Workbench Dashboard to have a domain for your companies internal use. Customers will typically use a domain like `<company_name>-ml-dashboard.com` but you are free to choose any domain you'd like.

!!!warning "Domains are tied to AWS Accounts" 
    When you create a new domain in AWS Route 53, that domain is tied to that AWS Account. You can do a cross account setup for domains but it's a bit more tricky. We recommend that each account where Workbench gets deployed owns the domain for that Dashboard.

### Multiple AWS Accounts
Many customers will have a dev/stage/prod set of AWS accounts, if that the case then the best practice is to make a domain specific to each account. So for instance:

- The AWS Dev Account gets: `<company_name>-ml-dashboard-dev.com` 
- The AWS Prod Account gets:  `<company_name>-ml-dashboard-prod.com`.

This means that when you go to that Dashboard it's super obvious which environment your on.

### Register the Domain

- **Open Route 53 Console** [Route 53 Console](https://console.aws.amazon.com/route53/)

- **Register your New Domain**
    - Click on **Registered domains** in the left navigation pane.
    - Click on **Register Domain**.
    - Enter your desired domain name and check for availability.
    - Follow the prompts to complete the domain registration process.
    - After registration, your domain will be listed under **Registered domains**.

## Request a SSL Certificate from ACM

1. **Open ACM Console:** [AWS Certificate Manager (ACM) Console](https://console.aws.amazon.com/acm/home)

1. **Request a Certificate:**
    - Click on **Request a certificate**.
    - Select **Request a public certificate** and click **Next**.

1. **Add Domain Names:**
    - Enter the domain name you registered (e.g., `yourdomain.com`).
    - Add any additional subdomains if needed (e.g., `www.yourdomain.com`).

1. **Validation Method:**
    - Choose **DNS validation** (recommended).
    - ACM will provide CNAME records that you need to add to your Route 53 hosted zone.

1. **Add Tags (Optional):**
    - Add any tags if you want to organize your resources.

1. **Review and Request:**
    - Review your request and click **Confirm and request**.

## Adding CNAME Records to Route 53

To complete the domain validation process for your SSL/TLS certificate, you need to add the CNAME records provided by AWS Certificate Manager (ACM) to your Route 53 hosted zone. This step ensures that you own the domain and allows ACM to issue the certificate.

### Finding CNAME Record Names and Values

You can find the CNAME record names and values in the AWS Certificate Manager (ACM) console:

1. **Open ACM Console:** [AWS Certificate Manager (ACM) Console](https://console.aws.amazon.com/acm/home)

2. **Select Your Certificate:**
    - Click on the certificate that is in the **Pending Validation** state.

3. **View Domains Section:**
    - Under the **Domains** section, you will see the CNAME record names and values that you need to add to your Route 53 hosted zone.

### Adding CName Records to Domain

1. **Open Route 53 Console:** [Route 53 Console](https://console.aws.amazon.com/route53/)

1. **Select Your Hosted Zone:**
    - Find and select the hosted zone for your domain (e.g., `yourdomain.com`).
    - Click on **Create record**.

1. **Add the First CNAME Record:**
     - For the **Record name**, enter the name provided by ACM (e.g., `_3e8623442477e9eeec.your-domain.com`). **Note:** they might already have `your-domain.com` next to this box, if so don't repeat it :)
     - For the **Record type**, select `CNAME`.
     - For the **Value**, enter the value provided by ACM (e.g., `_0908c89646d92.sdgjtdhdhz.acm-validations.aws.`) (include the trailing dot).
     - Leave the default settings for TTL.
     - Click on **Create records**.

1. **Add the Second CNAME Record:**
     - Repeat the process for the second CNAME record.
     - For the **Record name**, enter the second name provided by ACM (e.g., `_75cd9364c643caa.www.your-domain.com`).
     - For the **Record type**, select `CNAME`.
     - For the **Value**, enter the second value provided by ACM (e.g., `_f72f8cff4fb20f4.sdgjhdhz.acm-validations.aws.`)  (include the trailing dot).
     - Leave the default settings for TTL.
     - Click on **Create records**.

!!!tip "DNS Propagation and Cert Validation"
    After adding the CNAME records, these DNS records will propagate through the DNS system and ACM will automatically detect the validation records and validate the domain. This process can take a few minutes or up to an hour.

### Certificate States
After requesting a certificate, it will go through the following states:

- **Pending Validation**: The initial state after you request a certificate and before you complete the validation process. ACM is waiting for you to prove domain ownership by adding the CNAME records.

- **Issued**: This state indicates that the certificate has been successfully validated and issued. You can now use this certificate with your AWS resources.

- **Validation Timed Out**: If you do not complete the validation process within a specified period (usually 72 hours), the certificate request times out and enters this state.

- **Revoked**: This state indicates that the certificate has been revoked and is no longer valid.

- **Failed**: If the validation process fails for any reason, the certificate enters this state.

- **Inactive**: This state indicates that the certificate is not currently in use.

The certificate status should obviously be in the **Issued** state, if not please contact Workbench Support Team.

## Retrieving the Certificate ARN

1. **Open ACM Console:**
    - Go back to the [AWS Certificate Manager (ACM) Console](https://console.aws.amazon.com/acm/home).

2. **Check the Status:**
    - Once the CNAME records are added, ACM will automatically validate the domain.
    - Refresh the ACM console to see the updated status.
    - The status will change to "Issued" once validation is complete.

3. **Copy the Certificate ARN:**
    - Click on your issued certificate.
    - Copy the **Amazon Resource Name (ARN)** from the certificate details.

You now have the ARN for your certificate, which you can use in your AWS resources such as API Gateway, LoadBalancer, CloudFront, etc. Specifically for the Workbench-Dashboard Stack you will need to put this into your Workbench Config file when you deploy/update the stack.

```
"WORKBENCH_ROLE": "Workbench-ExecutionRole",
"WORKBENCH_CERTIFICATE_ARN": "arn:aws:acm:<region>:<account>:certificate/123-987-456-123-456789012",
```

### Update Route 53 to Point to New Load Balancer
If you deploy a new stack (new load balancer), you'll have to set up DNS 'A' records to point to that new load balancer.


#### Updating A Record in Route53
- Go to [Route 53 Console](https://console.aws.amazon.com/route53/)
- Click **Hosted zones** (on left panel)

#### New A Record(s)
- Create Record (orange button on right)
- Leave `subdomain` blank (for first A record)
- Click the **'Alias'** button (important)
- Routes traffic to 
    
    - Alias to Application and Classic Load Balanacer
    - AWS Region
    - Chooser Box (find LB domain)
  
- If you have another subdomain like `www.blah.com` then just go through the above steps again.

**Note:** The LB domain should looks something like `dualstack.workbe-workb-xyzabc-123456.us-west-2.elb.amazonaws.com`



#### Change A Record
- Click (or add) **A** records to point to your load balancer internal domain.
- Leave most of default options
- For **Route Traffic To**
    - Alias to Application and Classic Load Balancer
    - AWS Region
    - Chooser Box (find LB domain)

**Note:** The LB domain should looks something like `dualstack.workbe-workb-xyzabc-123456.us-west-2.elb.amazonaws.com`

 
## AWS Resources
- [AWS Adding or Changing DNS Records](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/resource-record-sets-creating.html)
- [AWS Certificate Manager (ACM) Documentation](https://docs.aws.amazon.com/acm/latest/userguide/acm-overview.html)
- [Requesting a Public Certificate](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html)
- [Validating Domain Ownership](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-validate-dns.html)
- [AWS Route 53 Documentation](https://docs.aws.amazon.com/route53/)
- [AWS API Gateway Documentation](https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html)

### Reference Materials

#### Finding the Load Balancer Internal Domain
**Note:** This is only for **NEW** A records. When you update an existing A Record, it should already 'find' this for you and present it as an option.

To find the internal domain of your load balancer in AWS:

1.	Go to the AWS Console:
	- Navigate to [EC2 Console](https://console.aws.amazon.com/route53/) (yes, Load Balancers are under EC2).

2.	Find the Load Balancer:
	- Under Load Balancing, click on Load Balancers.
  	- Look for the load balancer associated with your stack.
3.	Check the DNS Name:
	 -	Select the load balancer.
	 - In the Description tab, look for the DNS Name field. This is the internal domain you’re looking for (something like: Workbe-Workb-xyzabc-123456.us-west-2.elb.amazonaws.com).
	
4.	Use It for Your A Record:
	- Copy this DNS name and create the new A record in Route 53 or your DNS provider.


