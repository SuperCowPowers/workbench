# Domain and SSL Certificate Setup
!!!tip inline end "Need AWS Help?"
    The SuperCowPowers team is happy to give any assistance needed when setting up AWS and Workbench. So please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8)

The Workbench Dashboard serves over HTTPS, so it needs a domain name and an SSL certificate. Both must be in place **before** you deploy the [Dashboard Stack](dashboard_stack.md) — the stack reads the certificate ARN from your Workbench config.

## Which Option Fits You?
There are two ways to get there, and the right one depends on who controls your DNS. **Pick one** — you don't do both.

| | **Option 1: Company Subdomain** | **Option 2: Dedicated Route 53 Domain** |
|---|---|---|
| **Use when** | Your company already owns a domain managed by IT (GoDaddy, Cloudflare, corporate DNS) | Workbench gets its own domain, and this AWS account can own it |
| **Domain** | Already exists, managed outside AWS | Registered in Route 53 in this account |
| **Certificate** | Issued by your CA, **imported** into ACM | Requested from ACM, DNS-validated |
| **Dashboard DNS record** | **CNAME** in your company's DNS, created by your DNS admin | Route 53 **A** record, alias to the load balancer |
| **Renewal** | Manual — you re-import before expiry | Automatic |

Most teams land on **Option 1** — you already own a domain, and a subdomain under it is free. Option 2 is the tidier setup if you'd rather keep the Dashboard self-contained in one AWS account with certificate renewal handled for you.

Both options end the same way: a certificate ARN that goes into your Workbench config. Whichever you choose here, follow the matching section when you [point DNS at the load balancer](dashboard_stack.md#point-your-domain-at-the-load-balancer) after the stack is deployed.

---

## Option 1: Subdomain of Your Company Domain
Many companies already own a domain (e.g. `yourcompany.com`) registered with an external provider and managed by IT. In that case you don't register anything in Route 53 — you pick a subdomain for the Dashboard and import a certificate for it.

!!!warning "Don't create a Route 53 hosted zone for a domain you don't control"
    If your company's DNS lives elsewhere, creating a hosted zone for that domain in your AWS account does nothing — no nameservers delegate to it, so records you add there never resolve. Work with your DNS admin instead.

### Pick a Subdomain
Choose a name under a zone your DNS team controls, for example:

- `ml-dashboard.yourcompany.com`
- `workbench.dev.yourcompany.com`

If your team already has a wildcard certificate covering that level (e.g. `*.dev.yourcompany.com`), you can use it directly — skip to the import step below.

### Get the Certificate Files
Ask your DNS/IT team for the certificate for that name. You need three PEM-encoded pieces:

- **Certificate body** — the leaf certificate (`-----BEGIN CERTIFICATE-----`)
- **Private key** — the unencrypted key for that certificate (`-----BEGIN PRIVATE KEY-----`)
- **Certificate chain** — the intermediate CA certificates

### Import into ACM
Import in the **same region** as your Dashboard Stack.

Via the console:

1. **Open the [ACM Console](https://console.aws.amazon.com/acm/home)**
1. Click **Import certificate**
1. Paste the certificate body, private key, and chain into the three boxes
1. Click **Next**, add tags if you'd like, then **Import**

Or via the CLI:

```bash
aws acm import-certificate \
  --certificate fileb://certificate.pem \
  --private-key fileb://private-key.pem \
  --certificate-chain fileb://chain.pem \
  --region us-east-1
```

!!!warning "Imported certificates don't auto-renew"
    ACM renews certificates it issued, but not ones you import. Note the expiry date and re-import a fresh certificate before then — the Dashboard will start failing TLS the moment it lapses. ACM will send expiry notifications if you have them enabled.

An imported certificate goes straight to **Issued** — there are no validation CNAME records to add.

---

## Option 2: Dedicated Route 53 Domain

### Register the Domain
Customers typically use a domain like `<company_name>-ml-dashboard.com`, but you're free to choose any domain you'd like.

!!!warning "Domains are tied to AWS Accounts"
    A domain registered in Route 53 belongs to that AWS account. Cross-account domain setups are possible but tricky. We recommend that each account where Workbench is deployed owns the domain for its Dashboard.

If you have a dev/stage/prod set of AWS accounts, give each account its own domain:

- The AWS Dev Account gets: `<company_name>-ml-dashboard-dev.com`
- The AWS Prod Account gets: `<company_name>-ml-dashboard-prod.com`

When you're looking at a Dashboard, it's then obvious which environment you're on.

To register:

- **Open the [Route 53 Console](https://console.aws.amazon.com/route53/)**
- Click **Registered domains** in the left navigation pane
- Click **Register Domain**
- Enter your desired domain name and check availability
- Follow the prompts to complete registration

After registration, your domain will be listed under **Registered domains**.

### Request a Certificate from ACM

!!!note "Certificates are per-account"
    An ACM-issued public certificate can't be exported or shared across accounts. If you're standing up a Dashboard in a second account, request a new certificate there — two accounts can hold valid certificates for the same domain at the same time.

1. **Open the [ACM Console](https://console.aws.amazon.com/acm/home)**

1. **Request a Certificate:**
    - Click **Request a certificate**
    - Select **Request a public certificate** and click **Next**

1. **Add Domain Names:**
    - Enter the domain you registered (e.g. `yourdomain.com`)
    - Add any additional subdomains if needed (e.g. `www.yourdomain.com`)

1. **Validation Method:**
    - Choose **DNS validation** (recommended)
    - ACM will provide CNAME records to add to your Route 53 hosted zone

1. **Review and Request:**
    - Review your request and click **Confirm and request**

### Add the Validation CNAME Records
ACM issues the certificate once it can see a CNAME record proving you own the domain. You'll add one record per domain name on the certificate.

Find the names and values in the [ACM Console](https://console.aws.amazon.com/acm/home): click the certificate that's in the **Pending Validation** state, and look under the **Domains** section.

Then add them in Route 53:

1. **Open the [Route 53 Console](https://console.aws.amazon.com/route53/)**

1. Select the hosted zone for your domain (e.g. `yourdomain.com`) and click **Create record**

1. For each CNAME record ACM listed:
    - **Record name**: the name from ACM (e.g. `_3e8623442477e9eeec.your-domain.com`). **Note:** the console may already show `your-domain.com` next to the box — if so, don't repeat it :)
    - **Record type**: `CNAME`
    - **Value**: the value from ACM (e.g. `_0908c89646d92.sdgjtdhdhz.acm-validations.aws.`) — include the trailing dot
    - Leave the default TTL
    - Click **Create records**

!!!tip "DNS Propagation and Cert Validation"
    Once the CNAME records propagate, ACM detects them and validates the domain automatically. This takes a few minutes, occasionally up to an hour.

### Certificate States
A certificate moves through these states:

- **Pending Validation**: ACM is waiting for the CNAME records that prove domain ownership
- **Issued**: validated and ready to use
- **Validation Timed Out**: validation wasn't completed within 72 hours
- **Revoked**: the certificate has been revoked and is no longer valid
- **Failed**: validation failed
- **Inactive**: the certificate isn't currently in use

The certificate needs to be **Issued** before you deploy the Dashboard Stack. If it's stuck in another state, contact the Workbench Support Team.

---

## Copy the Certificate ARN into Your Config

1. In the [ACM Console](https://console.aws.amazon.com/acm/home), click your certificate
1. Copy the **Amazon Resource Name (ARN)** from the certificate details
1. Add it to your Workbench config file:

```json
"WORKBENCH_ROLE": "Workbench-ExecutionRole",
"WORKBENCH_CERTIFICATE_ARN": "arn:aws:acm:<region>:<account>:certificate/123-987-456-123-456789012",
```

You're now ready to deploy the [Dashboard Stack](dashboard_stack.md). After the stack comes up you'll point DNS at the new load balancer — those steps are on the Dashboard Stack page.

## AWS Resources
- [AWS Certificate Manager (ACM) Documentation](https://docs.aws.amazon.com/acm/latest/userguide/acm-overview.html)
- [Requesting a Public Certificate](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html)
- [Importing Certificates into ACM](https://docs.aws.amazon.com/acm/latest/userguide/import-certificate.html)
- [Validating Domain Ownership](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-validate-dns.html)
- [AWS Route 53 Documentation](https://docs.aws.amazon.com/route53/)
- [AWS Adding or Changing DNS Records](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/resource-record-sets-creating.html)
