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
import time # Added missing import for time.time()

matplotlib.use('Agg')

# --- Environment Variables ---
# These must be set in your Snowflake Service Specification YAML (as env vars or secrets)
# For the main SPCS DB connection
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

# For the CortexChat class API calls (requires JWT authentication)
USER = os.getenv("USER")
ACCOUNT = os.getenv("ACCOUNT") # Note: This might be the same as SNOWFLAKE_ACCOUNT
RSA_PRIVATE_KEY_PATH = os.getenv("RSA_PRIVATE_KEY_PATH") # Path to key file INSIDE the container
AGENT_ENDPOINT = f"https://{SNOWFLAKE_HOST}/api/v2/cortex/agent:run"
# AGENT_ENDPOINT = f"https://{SNOWFLAKE_HOST}/api/v2/cortex/analyst/message"
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
MODEL = os.getenv("MODEL")

DEBUG = True

# Initializes app
app = App(token=SLACK_BOT_TOKEN)

def get_logger():
    logger = logging.getLogger("cortex-analyst-slack-spcs")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

log = get_logger()

# FIX 1: Moved plot_chart to the global scope so it's defined before it is called.
def plot_chart(df):
    plt.figure(figsize=(10, 6), facecolor='#333333')
    plt.pie(df[df.columns[1]], 
            labels=df[df.columns[0]], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=['#1f77b4', '#ff7f0e'], 
            textprops={'color':"white",'fontsize': 16})
    plt.axis('equal')
    plt.gca().set_facecolor('#333333')   
    plt.tight_layout()
    file_path_jpg = 'pie_chart.jpg'
    plt.savefig(file_path_jpg, format='jpg')
    file_size = os.path.getsize(file_path_jpg)
    file_upload_url_response = app.client.files_getUploadURLExternal(filename=file_path_jpg,length=file_size)
    if DEBUG:
        print(file_upload_url_response)
    file_upload_url = file_upload_url_response['upload_url']
    file_id = file_upload_url_response['file_id']
    with open(file_path_jpg, 'rb') as f:
        response = requests.post(file_upload_url, files={'file': f})
    img_url = None
    if response.status_code != 200:
        print("File upload failed", response.text)
    else:
        response = app.client.files_completeUploadExternal(files=[{"id":file_id, "title":"chart"}])
        if DEBUG:
            print(response)
        img_url = response['files'][0]['permalink']
        time.sleep(2)
    return img_url

@app.event("message")
def handle_message_events(ack, body, say):
    try:
        ack()
        prompt = body['event']['text']
        say(
            text = "Snowflake Cortex AI is generating a response",
            blocks=[
                {"type": "divider"},
                {"type": "section", "text": {"type": "plain_text", "text": ":snowflake: Snowflake Cortex AI is generating a response. Please wait..."}},
                {"type": "divider"},
            ]
        )
        show_sql = False
        if prompt.strip().endswith("/sql"):
            show_sql = True
            prompt = prompt.strip()[:-4].strip()
        response = ask_agent(prompt)
        display_agent_response(response, say, show_sql=show_sql)
    except Exception as e:
        error_info = f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
        log.error(f"Error in handle_message_events: {error_info}")
        say(
            text = "Request failed...",
            blocks=[
                {"type": "divider"},
                {"type": "section", "text": {"type": "plain_text", "text": f"An error occurred: {e}"}},
                {"type": "divider"},
            ]
        )        

def ask_agent(prompt):
    resp = CORTEX_APP.chat(prompt)
    return resp

def display_agent_response(content, say, show_sql=False):
    has_displayed_primary_content = False

    # 1. Handle SQL content (if any)
    if content.get('sql'):
        sql = content['sql']
        df = None
        sql_execution_error = None
        try:
            df = pd.read_sql(sql, CONN)
        except Exception as e:
            sql_execution_error = f"Error executing SQL: {type(e).__name__}: {e}"
            print(sql_execution_error)

        # Prepare blocks for SQL and its results/error
        sql_blocks_elements = [
            {
                "type": "rich_text_quote", # Label for the whole section
                "elements": [{"type": "text", "text": "Analyst Response:", "style": {"bold": True}}]
            },
        ]
        if show_sql:
            sql_blocks_elements.append({
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": f"SQL:\n{sql}"}]
            })
        if df is not None:
            sql_blocks_elements.append({
                "type": "rich_text_preformatted", # DataFrame results
                "elements": [{"type": "text", "text": f"\nResults:\n{df.to_string()}"}]
            })
        elif sql_execution_error:
            sql_blocks_elements.append({
                "type": "rich_text_preformatted", # SQL execution error
                "elements": [{"type": "text", "text": f"\nError:\n{sql_execution_error}"}]
            })
        
        say(
            text="SQL Query and Results",
            blocks=[{"type": "rich_text", "elements": sql_blocks_elements}]
        )

        if df is not None and len(df.columns) > 1:
            chart_img_url = None # Reset for safety
            try:
                chart_img_url = plot_chart(df) # plot_chart should handle its own errors or return None
            except Exception as e:
                chart_error_info = f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
                print(f"Warning: Data likely not suitable for displaying as a chart. {chart_error_info}")
            if chart_img_url:
                say(
                    text = "Chart",
                    blocks=[{
                        "type": "image", "title": {"type": "plain_text", "text": "Chart"},
                        "block_id": "image_" + str(time.time()), # Unique block_id
                        "image_url": chart_img_url, "alt_text": "Generated Chart"
                    }]
                )
        has_displayed_primary_content = True

    # 2. Handle Suggestions (if any)
    # This can be displayed even if SQL was present, or if SQL was not.
    # The `content['text']` here is the one prepared by _parse_response,
    # potentially the text that accompanied the suggestions.
    if content.get('suggestions'):
        suggestions_list = content['suggestions']
        
        suggestion_display_blocks = []
        
        # Add the introductory text if it exists and is relevant (especially if no SQL was shown)
        if content.get('text'):
             suggestion_display_blocks.append({
                "type": "rich_text_section", # Or preformatted if it contains newlines
                "elements": [{"type": "text", "text": f"{content['text']}"}]
            })

        # Format suggestions as a list
        formatted_suggestions = "\n".join([f"• {s}" for s in suggestions_list])
        suggestion_display_blocks.append({
            "type": "rich_text_quote",
            "elements": [{"type": "text", "text": "\nSuggestions:", "style": {"bold": True}}]
        })
        suggestion_display_blocks.append({
            "type": "rich_text_preformatted",
            "elements": [{"type": "text", "text": formatted_suggestions}]
        })

        say(
            text="Suggestions available",
            blocks=[{"type": "rich_text", "elements": suggestion_display_blocks}]
        )
        has_displayed_primary_content = True # Mark that suggestions were primary or additional content

    # 3. Handle plain text responses (if no SQL and no suggestions were found/displayed)
    elif not has_displayed_primary_content and content.get('text'):
        # This 'text' is the general LLM response if no tools (SQL/suggestions) were effectively used or parsed.
        say(
            text="Answer:",
            blocks=[{
                "type": "rich_text",
                "elements": [{
                    "type": "rich_text_quote",
                    "elements": [{"type": "text", "text": f"Answer: {content['text']}", "style": {"bold": True}}]
                }]
            }]
        )
        has_displayed_primary_content = True

    # 4. If nothing meaningful was found in content to display
    if not has_displayed_primary_content:
        say(text="I received a response, but couldn't find specific information to display.")
        print(f"Debug: Unhandled content structure in display_agent_response: {content}")

def init():
    log.info("SPCS environment detected. Attempting connection with OAuth token.")

    if not os.path.exists(SPCS_TOKEN_FILE):
        log.critical(f"SPCS token file not found at {SPCS_TOKEN_FILE}. This container can only run in SPCS.")
        raise FileNotFoundError(f"SPCS token file not found at {SPCS_TOKEN_FILE}.")

    with open(SPCS_TOKEN_FILE, 'r') as f:
        token = f.read()

    if not all([SNOWFLAKE_ACCOUNT, SNOWFLAKE_HOST, WAREHOUSE, ROLE, token]):
        missing_vars = [var for var, val in {
            "SNOWFLAKE_ACCOUNT": SNOWFLAKE_ACCOUNT, "SNOWFLAKE_HOST": SNOWFLAKE_HOST,
            "WAREHOUSE": WAREHOUSE, "ROLE": ROLE, "SPCS Token": token
        }.items() if not val]
        log.critical(f"Missing required SPCS connection variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required SPCS connection variables: {', '.join(missing_vars)}")

    try:
        conn = snowflake.connector.connect(
            authenticator='oauth',
            token=token,
            account=SNOWFLAKE_ACCOUNT,
            host=SNOWFLAKE_HOST,
            warehouse=WAREHOUSE,
            role=ROLE,
            database=DATABASE,
            schema=SCHEMA
        )
        log.info("Successfully connected to Snowflake using SPCS OAuth token.")
    except Exception as e:
        log.critical(f"Fatal error connecting to Snowflake using SPCS OAuth token: {e}")
        raise

    # Setup CortexChat Instance - This still requires JWT credentials for its own API calls.
    cortex_vars = [AGENT_ENDPOINT, SEMANTIC_MODEL, MODEL, token]
    if not all(cortex_vars):
        missing = [name for name, val in zip(
            ["AGENT_ENDPOINT", "SEMANTIC_MODEL", "MODEL", "SPCS Token"],
            cortex_vars
        ) if not val]
        raise ValueError(f"Missing variables for CortexChat: {missing}")

    cortex_app_instance = cortex_chat_docker.CortexChat(
        agent_url=AGENT_ENDPOINT,
        semantic_model=SEMANTIC_MODEL,
        model=MODEL,
        logger=log,
    )
    log.info(">>>>>>>>>> Init complete (Snowflake connection and CortexChat).")
    return conn, cortex_app_instance

# Slack Socket Mode App
def start_slack():
    global CONN, CORTEX_APP
    CONN, CORTEX_APP = init()
    log.info("Starting Slack SocketModeHandler...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()

# Dummy Flask App for Snowflake health check
health_app = Flask(__name__)

@health_app.get("/healthcheck")
def readiness_probe():
    return "I'm ready!"

if __name__ == "__main__":
    # Start dummy HTTP server in a thread
    Thread(target=lambda: health_app.run(host="0.0.0.0", port=8080)).start()
    
    # Run Slack bot in main thread
    start_slack()
