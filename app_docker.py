"""
This module implements a Slack bot that interacts with the Snowflake Cortex API.

This script sets up a Slack bot that listens for messages, sends them to the
Cortex API, and then formats and displays the response in Slack. It can handle
text, SQL queries, and image charts.
"""

import os
import logging
import sys
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import snowflake.connector
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cortex_chat_docker
from threading import Thread
from flask import Flask
import time
from slack_sdk.errors import SlackApiError
from connection_manager import SnowflakeConnectionManager
import requests

matplotlib.use("Agg")

# --- Environment Variables ---
# These must be set in your Snowflake Service Specification YAML (as env vars or secrets)
SPCS_TOKEN_FILE = "/snowflake/session/token"
SNOWFLAKE_HOST = os.getenv("SNOWFLAKE_HOST")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
DATABASE = os.getenv("DATABASE")
SCHEMA = os.getenv("SCHEMA")
WAREHOUSE = os.getenv("WAREHOUSE")
ROLE = os.getenv("ROLE")

# For the Slack Bot
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

# For the CortexChat class API calls
USER = os.getenv("USER")
ACCOUNT = os.getenv("ACCOUNT")
AGENT_ENDPOINT = f"https://{SNOWFLAKE_HOST}/api/v2/cortex/agent:run"
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
MODEL = os.getenv("MODEL")

DEBUG = True

# Initializes the Slack app
app = App(token=SLACK_BOT_TOKEN)
messages = []


def get_logger():
    """Initializes and returns a logger instance."""
    logger = logging.getLogger("cortex-analyst-slack-spcs")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


log = get_logger()
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def send_dataframe_as_csv(df, channel_id, initial_comment, prompt):
    """
    Saves a DataFrame to a CSV file and uploads it to Slack.

    Args:
        df: The pandas DataFrame to save.
        channel_id: The ID of the Slack channel to upload the file to.
        initial_comment: The initial comment to post with the file.
        prompt: The original user prompt that generated the DataFrame.
    """
    if df.empty:
        app.client.chat_postMessage(
            channel=channel_id,
            text=f"Your query '{prompt}' ran successfully but returned no data to create a file.",
        )
        return
    try:
        csv_file_name = f"cortex_results.csv"
        df.to_csv(csv_file_name, index=False)
        app.client.files_upload_v2(
            channel=channel_id,
            file=csv_file_name,
            title=f"Results for: {prompt}",
            initial_comment=initial_comment,
        )
        log.info(f"Successfully uploaded {csv_file_name} to channel {channel_id}")
    except Exception as e:
        error_msg = (
            f"Sorry, I ran into an error while creating or uploading the CSV file: {e}"
        )
        log.error(error_msg)
        app.client.chat_postMessage(channel=channel_id, text=error_msg)
    finally:
        # Clean up the local file after uploading
        if os.path.exists(csv_file_name):
            os.remove(csv_file_name)


@app.event("message")
def handle_message_events(ack, body, say):
    """
    Handles incoming messages from Slack.

    This function is triggered when a message is posted in a channel the bot is in.
    It parses the message for commands, sends the prompt to the Cortex API, and
    displays the response.
    """
    try:
        ack()
        original_prompt = body["event"]["text"]
        channel_id = body["event"]["channel"]

        clean_prompt = original_prompt
        commands = {"/sql": False, "/img": False}

        for cmd, _ in commands.items():
            if cmd in clean_prompt:
                commands[cmd] = True
                clean_prompt = clean_prompt.replace(
                    cmd, ""
                )  # Remove command from prompt

        clean_prompt = clean_prompt.strip()

        # Set flags from the parsed commands
        show_sql = commands["/sql"]
        show_img = commands["/img"]

        say(
            text="Snowflake Cortex AI is generating a response",
            blocks=[
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": ":snowflake: Snowflake Cortex AI is generating a response. Please wait...",
                    },
                },
                {"type": "divider"},
            ],
        )

        response = ask_agent(clean_prompt)
        display_agent_response(
            response, say, channel_id, show_sql, show_img, original_prompt
        )
    except Exception as e:
        error_info = (
            f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
        )
        log.error(error_info)
        say(
            text="Request failed...",
            blocks=[
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": f"{error_info}",
                    },
                },
                {"type": "divider"},
            ],
        )


def ask_agent(prompt):
    """Sends a prompt to the Cortex API and returns the response."""
    resp = CORTEX_APP.chat(prompt)
    return resp


def display_agent_response(content, say, channel_id, show_sql, show_img, prompt):
    """
    Displays the agent's response in Slack.

    This function formats the response from the Cortex API and displays it in
    Slack. It can handle text, SQL, and suggestions.
    """
    has_displayed_primary_content = False

    # Handle SQL content
    if content.get("sql"):
        sql = content["sql"]
        df = None
        sql_execution_error = None

        try:
            # Get a fresh, valid connection from the manager
            conn_from_manager = CONN_MANAGER.get_connection()
            df = pd.read_sql(sql, conn_from_manager)
        except Exception as e:
            sql_execution_error = f"Error executing SQL: {type(e).__name__}: {e}"
            log.error(sql_execution_error)

        # Prepare blocks for the message
        sql_blocks_elements = [
            {
                "type": "rich_text_quote",
                "elements": [
                    {
                        "type": "text",
                        "text": "Analyst Response:",
                        "style": {"bold": True},
                    }
                ],
            }
        ]

        if show_sql:
            sql_blocks_elements.append(
                {
                    "type": "rich_text_preformatted",
                    "elements": [{"type": "text", "text": f"SQL:\n{sql}"}],
                }
            )

        if df is not None:
            sql_blocks_elements.append(
                {
                    "type": "rich_text_preformatted",
                    "elements": [
                        {"type": "text", "text": f"\nResults:\n{df.to_string()}"}
                    ],
                }
            )
        elif sql_execution_error:
            sql_blocks_elements.append(
                {
                    "type": "rich_text_preformatted",
                    "elements": [
                        {"type": "text", "text": f"\nError:\n{sql_execution_error}"}
                    ],
                }
            )

        try:
            # Try to send the results as a normal message
            say(
                text="SQL Query and Results",
                blocks=[{"type": "rich_text", "elements": sql_blocks_elements}],
            )

        except SlackApiError as e:
            if e.response["error"] == "msg_blocks_too_long":
                log.info("Message too long. Handling with fallback...")
                file_comment = ""
                if show_sql:
                    # Send a separate message with just the SQL query
                    say(
                        text=f"SQL for your query: {prompt}",
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"The results for your query were too long to display, but here is the SQL you requested:",
                                },
                            },
                            {
                                "type": "rich_text",
                                "elements": [
                                    {
                                        "type": "rich_text_preformatted",
                                        "elements": [{"type": "text", "text": sql}],
                                    }
                                ],
                            },
                        ],
                    )
                    file_comment = "Here are the full results in the attached CSV file."
                else:
                    file_comment = f"The results for your query, '{prompt}', were too long to display directly. Here they are in the attached file."
                send_dataframe_as_csv(df, channel_id, file_comment, prompt)
            else:
                log.error(f"An unexpected Slack API error occurred: {e}")
                say(
                    text=f"Sorry, I couldn't send the response due to a Slack error: `{e.response['error']}`"
                )

        if df is not None and show_img and len(df.columns) > 1:
            plot_chart(df, channel_id, prompt, say)

        has_displayed_primary_content = True

    # Handle Suggestions
    if content.get("suggestions"):
        suggestions_list = content["suggestions"]

        suggestion_display_blocks = []

        if content.get("text"):
            suggestion_display_blocks.append(
                {
                    "type": "rich_text_section",
                    "elements": [{"type": "text", "text": f"{content['text']}"}],
                }
            )

        # Format suggestions as a list
        formatted_suggestions = "\n".join([f"• {s}" for s in suggestions_list])
        suggestion_display_blocks.append(
            {
                "type": "rich_text_quote",
                "elements": [
                    {"type": "text", "text": "\nSuggestions:", "style": {"bold": True}}
                ],
            }
        )
        suggestion_display_blocks.append(
            {
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": formatted_suggestions}],
            }
        )

        say(
            text="Suggestions available",
            blocks=[{"type": "rich_text", "elements": suggestion_display_blocks}],
        )
        has_displayed_primary_content = True

    # Handle plain text responses
    elif not has_displayed_primary_content and content.get("text"):
        say(
            text="Answer:",
            blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": content["text"]}}
            ],
        )
        has_displayed_primary_content = True

    # If nothing meaningful was found in content to display
    if not has_displayed_primary_content:
        say(
            text="I received a response, but couldn't find specific information to display."
        )
        log.debug(f"Unhandled content structure in display_agent_response: {content}")


def plot_chart(df, channel_id, prompt, say):
    """
    Analyzes the DataFrame and generates the most appropriate plot, then posts it to Slack.

    Args:
        df: The pandas DataFrame to plot.
        channel_id: The ID of the Slack channel to upload the plot to.
        prompt: The original user prompt that generated the DataFrame.
        say: The Slack `say` function.
    """
    log.info(f"Attempting to generate a dynamic chart for the prompt: '{prompt}'")

    # Convert date-like columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
                log.info(f"Successfully converted column '{col}' to datetime.")
            except (ValueError, TypeError):
                pass

    plot_created = False

    # Analyze DataFrame columns to choose a plot type
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    datetime_cols = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    string_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(12, 7))

    # Rule 1: Time-Series Line Plot
    if len(datetime_cols) == 1 and len(numeric_cols) >= 1:
        log.debug("Rule matched: Time-Series. Creating Line Plot.")
        x_ax, y_ax = datetime_cols[0], numeric_cols[0]
        plt.plot(df[x_ax], df[y_ax], marker="o", linestyle="-")
        plt.xlabel(x_ax.replace("_", " ").title())
        plt.ylabel(y_ax.replace("_", " ").title())
        plt.title(f"Time-Series Analysis: {y_ax.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
        plot_created = True

    # Rule 2: Pie Chart for few categories
    elif (
        len(string_cols) == 1
        and len(numeric_cols) == 1
        and (2 <= df[string_cols[0]].nunique() <= 7)
    ):
        log.debug("Rule matched: Few Categories. Creating Pie Chart.")
        labels_col, values_col = string_cols[0], numeric_cols[0]
        plt.pie(
            df[values_col], labels=df[labels_col], autopct="%1.1f%%", startangle=140
        )
        plt.title(
            f"Distribution of {values_col.replace('_', ' ').title()} by {labels_col.replace('_', ' ').title()}"
        )
        plt.axis("equal")
        plot_created = True

    # Rule 3: Bar Chart for categorical data
    elif len(string_cols) == 1 and len(numeric_cols) >= 1:
        log.debug("Rule matched: Categorical. Creating Bar Chart.")
        x_ax, y_ax = string_cols[0], numeric_cols[0]
        plt.bar(df[x_ax], df[y_ax])
        plt.xlabel(x_ax.replace("_", " ").title())
        plt.ylabel(y_ax.replace("_", " ").title())
        plt.title(
            f"{y_ax.replace('_', ' ').title()} by {x_ax.replace('_', ' ').title()}"
        )
        plt.xticks(rotation=45, ha="right")
        plot_created = True

    # Rule 4: Scatter Plot for two numeric columns
    elif len(numeric_cols) >= 2:
        log.debug("Rule matched: Two Numerics. Creating Scatter Plot.")
        x_ax, y_ax = numeric_cols[0], numeric_cols[1]
        plt.scatter(df[x_ax], df[y_ax])
        plt.xlabel(x_ax.replace("_", " ").title())
        plt.ylabel(y_ax.replace("_", " ").title())
        plt.title(
            f"Relationship between {x_ax.replace('_', ' ').title()} and {y_ax.replace('_', ' ').title()}"
        )
        plot_created = True

    # Save and upload the plot if one was created
    if plot_created:
        plt.tight_layout()
        file_path_jpg = "chart.jpg"
        plt.savefig(file_path_jpg, format="jpg")
        plt.close()

        try:
            app.client.files_upload_v2(
                channel=channel_id,
                file=file_path_jpg,
                title=f"Chart for: {prompt}",
                initial_comment="Here is a chart generated from your query.",
            )
            log.info(
                f"Successfully uploaded chart for prompt '{prompt}' to channel {channel_id}"
            )
        except SlackApiError as e:
            log.error(f"Failed to upload chart to Slack: {e.response['error']}")
            app.client.chat_postMessage(
                channel=channel_id,
                text=f"Sorry, I couldn't upload the chart. Error: `{e.response['error']}`",
            )
        finally:
            if os.path.exists(file_path_jpg):
                os.remove(file_path_jpg)
    else:
        log.warning(
            "Could not determine a suitable chart type for the given data. No chart was created."
        )
        say(
            text="I analyzed the data but couldn't determine a suitable chart type to create."
        )


def init():
    """Initializes the Snowflake connection and CortexChat instance."""
    log.info("SPCS environment detected. Initializing with OAuth token...")

    if not os.path.exists(SPCS_TOKEN_FILE):
        raise FileNotFoundError(f"SPCS token file not found at {SPCS_TOKEN_FILE}.")
    with open(SPCS_TOKEN_FILE, "r") as f:
        token = f.read()

    try:
        initial_conn = snowflake.connector.connect(
            authenticator="oauth",
            token=token,
            account=SNOWFLAKE_ACCOUNT,
            host=SNOWFLAKE_HOST,
            warehouse=WAREHOUSE,
            role=ROLE,
            database=DATABASE,
            schema=SCHEMA,
        )
        log.info("Initial Snowflake OAuth connection for pandas queries successful.")
    except Exception as e:
        log.critical(f"Fatal error creating initial Snowflake connection: {e}")
        raise

    # Initialize the Connection Manager
    conn_manager = SnowflakeConnectionManager(
        logger=log, initial_connection=initial_conn
    )
    log.info("Connection Manager initialized.")

    # Initialize CortexChat
    log.info("Initializing CortexChat instance...")
    cortex_app_instance = cortex_chat_docker.CortexChat(
        agent_url=AGENT_ENDPOINT, semantic_model=SEMANTIC_MODEL, model=MODEL, logger=log
    )

    log.info(">>>>>>>>>> Init complete.")
    return conn_manager, cortex_app_instance


def start_slack():
    """Starts the Slack bot."""
    global CONN_MANAGER, CORTEX_APP
    CONN_MANAGER, CORTEX_APP = init()
    log.info("Starting Slack SocketModeHandler...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()


# Dummy Flask App for Snowflake health check
health_app = Flask(__name__)


@health_app.get("/healthcheck")
def readiness_probe():
    """A simple health check endpoint for the container service."""
    return "I'm ready!"


if __name__ == "__main__":
    # Start the dummy HTTP server in a separate thread
    Thread(target=lambda: health_app.run(host="0.0.0.0", port=8000)).start()

    # Run the Slack bot in the main thread
    start_slack()
