# Setting Up AWS Client VPN

Follow the steps below to set up and connect using AWS Client VPN.

## Step 1: Create a Client VPN Endpoint in AWS

1. Go to the [VPC Dashboard](https://console.aws.amazon.com/vpc) in the AWS Management Console.
2. Select **Client VPN Endpoints** and click **Create Client VPN Endpoint**.
3. Fill in the required details:
   - **Client IPv4 CIDR**: Choose an IP range (e.g., `10.0.0.0/22`) that doesn’t overlap with your VPC CIDR.
   - **Server Certificate ARN**: Use or create an SSL certificate using AWS Certificate Manager (ACM).
   - **Authentication Options**: Use either **Mutual Authentication** (client certificates) or **Active Directory** (for user-based authentication).
   - **Connection Log Options**: Optional; you can set up CloudWatch logs.
4. Click **Create Client VPN Endpoint**.

## Step 2: Associate the Client VPN Endpoint with a VPC Subnet

1. Once the endpoint is created, select it and click on **Associations**.
2. Choose a **Target Network Association** (a subnet in the VPC where you want VPN clients to access).
3. This allows traffic from the VPN clients to route through your chosen VPC subnet.

## Step 3: Authorize VPN Clients to Access the VPC

1. Under the **Authorization Rules** tab, click **Add Authorization Rule**.
2. For **Destination network**, specify `0.0.0.0/0` to allow access to all resources in the VPC.
3. Set **Grant access to** to `Allow access` and specify the group you created or allow all users.

## Step 4: Download and Install AWS VPN Client

1. [Download the AWS Client VPN](https://aws.amazon.com/vpn/client-vpn-download/) for macOS.
2. Install the client on your Mac.

## Step 5: Download the Client Configuration File

1. In the AWS Console, go to your Client VPN Endpoint.
2. Click on **Download Client Configuration**. This file contains the connection details required by the VPN client.

## Step 6: Import the Configuration File and Connect

1. Open the AWS Client VPN app on your Mac.
2. Click **File** -> **Manage Profiles** -> **Add Profile**.
3. Import the configuration file you downloaded.
4. Enter your credentials if required (depending on the authentication method you chose).
5. Click **Connect**.

## Benefits of Using AWS Client VPN

- **Simple Setup**: Minimal steps and no need for additional infrastructure.
- **Secure**: Uses TLS to secure connections, and you control who has access.
- **Direct Access**: Provides direct access to your AWS resources, including the Redis cluster.

## Conclusion

AWS Client VPN is a straightforward, secure, and effective solution for connecting your laptop to an AWS VPC. It requires minimal setup and provides all the security controls you need, making it ideal for a single laptop and user.