# tldr-web-app 
tldr-web-app project explores patent summarization and research paper abstract generation using large language models.

## Description
The project focuses on harnessing the power of large language models to create precise summaries that distill essential insights from academic and legal documents. We have built a cloud-based web application that generates abstracts with minimal response time and offers users a customized experience. We have integrated the application with a Large Language Model (LLM) for inference. The app further considers security, scalability, and availability, ensuring a reliable and seamless platform for generating abstracts and summaries.

## Architecture

<img src="/img/architecture_diagram.png" width="65%">

In our project's backend infrastructure, we utilized AWS Lambda for on-demand and scalable compute services, allowing us to divide backend code into separate Lambda functions for enhanced scalability, customization, and security. Our deployment procedure for Lambda functions involved leveraging AWS S3, an object storage service known for its secure and robust data storage capabilities. Normally, AWS lambda has a limitation of supporting a maximum of 250 MB of uncompressed code repository storage size. To work around Lambda’s size constraints, especially when dealing with large, multi-gigabyte models like the fine-tuned LLM, we adopted a Dockerization strategy. This entailed creating Docker images locally and uploading them to AWS ECR, subsequently facilitating their use within Lambda functions. AWS API Gateway managed frontend-backend communication via token authentication. AWS Cognito handled user authentication and session management, bolstering security. Summaries and abstracts were securely stored in AWS DynamoDB. For diagnostics, AWS CloudWatch monitored resources. AWS IAM managed permissions, while AWS SageMaker provided computational capabilities. Firebase hosted the app, offering real-time data updates, authentication, and streamlined deployment.
