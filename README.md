# Cortex Analyst on Slack Setup Guide

This guide outlines the necessary steps and configurations to run the Cortex Analyst on Slack.

## 1. Slack Custom App Setup

To begin, you need to set up a custom application within Slack. Follow the comprehensive tutorial provided by Snowflake:

[Integrate Snowflake Cortex Agents with Slack Quickstart](https://quickstarts.snowflake.com/guide/integrate_snowflake_cortex_agents_with_slack/index.html?index=..%2F..index#1)

## 2. Snowflake Authentication

Configure a key pair authentication key to connect to Snowflake securely.

## 3. Environment Variables (`.env` file)

Once your Slack app is configured and installed in a Slack channel, create a `.env` file in the root directory of your project with the following information. Replace the placeholder empty strings with your actual credentials and details:

DEMO_DATABASE=''
DEMO_SCHEMA=''
WAREHOUSE=''
DEMO_USER=''
DEMO_USER_ROLE=''
SEMANTIC_MODEL=''
SEARCH_SERVICE=''
ACCOUNT=''
HOST=''
AGENT_ENDPOINT=''
SLACK_APP_TOKEN=''
SLACK_BOT_TOKEN=''
RSA_PRIVATE_KEY_PATH='rsa_key.p8'
MODEL = ''

## 4. Running the Message Broker

There are three different ways to run the message broker:

### A. Locally

1.  Run the Python application directly:
    ```bash
    python3 app_local.py
    ```

### B. Docker (Local or Remote)

2.  Create a Docker image using the provided `Dockerfile` and run the container.

### C. Snowpark Container Services (SPCS)

3.  To run the service on SPCS, create a Docker image with the `Dockerfile_SCPS` provided and then run the container.

    **SPCS Service Configuration:**
    Before deploying to SPCS, you need to configure the service. You can use the following tutorial as a guide:

    [Introduction to Snowpark Container Services Quickstart](https://quickstarts.snowflake.com/guide/intro_to_snowpark_container_services/index.html#0)

    Use the following command to create and run the service. **Remember to fill in the environment variables** within the `env:` section:

    ```sql
    CREATE SERVICE service_name
      IN COMPUTE POOL compute_pool_name
      FROM SPECIFICATION $$
    spec:
      containers:
        - name: cortex
          image: /db/schema/image_repository/image_name:latest
          env:
            SERVER_PORT: 8000
            ACCOUNT: ''
            HOST: ''
            USER: ''
            DATABASE: ''
            SCHEMA: ''
            ROLE: ''
            WAREHOUSE: ''
            SLACK_APP_TOKEN: ''
            SLACK_BOT_TOKEN: ''
            SEMANTIC_MODEL: ''
            RSA_PRIVATE_KEY_PATH: ''
            MODEL: ""
          readinessProbe:
            port: 8000
            path: /healthcheck
      endpoints:
        - name: cortexendpoint
          port: 8000
          public: true
    $$
      EXTERNAL_ACCESS_INTEGRATIONS = (slack_integration)
      MIN_INSTANCES = 1
      MAX_INSTANCES = 2;
    ```
