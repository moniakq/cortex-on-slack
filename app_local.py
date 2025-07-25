from typing import Any
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import snowflake.connector
import pandas as pd
from snowflake.core import Root
from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt 
from snowflake.snowpark import Session
import numpy as np
import cortex_chat
import time
import requests
from slack_sdk.errors import SlackApiError

matplotlib.use('Agg')
load_dotenv()

ACCOUNT = os.getenv("ACCOUNT")
HOST = os.getenv("HOST")
USER = os.getenv("DEMO_USER")
DATABASE = os.getenv("DEMO_DATABASE")
SCHEMA = os.getenv("DEMO_SCHEMA")
ROLE = os.getenv("DEMO_USER_ROLE")
WAREHOUSE = os.getenv("WAREHOUSE")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT")
SEMANTIC_MODEL = os.getenv("SEMANTIC_MODEL")
SEARCH_SERVICE = os.getenv("SEARCH_SERVICE")
RSA_PRIVATE_KEY_PATH = os.getenv("RSA_PRIVATE_KEY_PATH")
MODEL = os.getenv("MODEL")

DEBUG = True

# Initializes app
app = App(token=SLACK_BOT_TOKEN)
messages = []

def send_dataframe_as_csv(df, channel_id, initial_comment, prompt):
    """Saves a DataFrame to a CSV and uploads it to Slack."""
    if df.empty:
        app.client.chat_postMessage(
            channel=channel_id,
            text=f"Your query '{prompt}' ran successfully but returned no data to create a file."
        )
        return

    try:
        csv_file_name = f"cortex_results.csv"
        df.to_csv(csv_file_name, index=False)
        
        app.client.files_upload_v2(
            channel=channel_id,
            file=csv_file_name,
            title=f"Results for: {prompt}",
            initial_comment=initial_comment
        )
        print(f"Successfully uploaded {csv_file_name} to channel {channel_id}")
    except Exception as e:
        error_msg = f"Sorry, I ran into an error while creating or uploading the CSV file: {e}"
        print(error_msg)
        app.client.chat_postMessage(channel=channel_id, text=error_msg)
    finally:
        # Clean up the local file after uploading
        if os.path.exists(csv_file_name):
            os.remove(csv_file_name)

@app.event("message")
def handle_message_events(ack, body, say):
    try:
        ack()
        prompt = body['event']['text']
        channel_id = body['event']['channel']
        say(
            text = "Snowflake Cortex AI is generating a response",
            blocks=[
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": ":snowflake: Snowflake Cortex AI is generating a response. Please wait...",
                    }
                },
                {
                    "type": "divider"
                },
            ]
        )
        # 1. Initialize a flag to control SQL visibility
        show_sql = False
        # 2. Check for the /sql command
        if prompt.strip().endswith("/sql"):
            show_sql = True
            prompt = prompt.strip()[:-4].strip()
        response = ask_agent(prompt)
        # print(prompt)
        # print(response['text'])
        # print(response['sql'])
        display_agent_response(response, say, channel_id, show_sql, prompt)
    except Exception as e:
        error_info = f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
        print(error_info)
        say(
            text = "Request failed...",
            blocks=[
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": f"{error_info}",
                    }
                },
                {
                    "type": "divider"
                },
            ]
        )        

def ask_agent(prompt):
    resp = CORTEX_APP.chat(prompt)
    return resp

def display_agent_response(content, say, channel_id, show_sql, prompt):
    # This function is now much cleaner
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

        # Prepare blocks for the message
        sql_blocks_elements = [
            {"type": "rich_text_quote", "elements": [{"type": "text", "text": "Analyst Response:", "style": {"bold": True}}]}
        ]
        
        if show_sql:
            sql_blocks_elements.append({"type": "rich_text_preformatted", "elements": [{"type": "text", "text": f"SQL:\n{sql}"}]})

        if df is not None:
            # Add the results to the blocks list. This might be very long.
            sql_blocks_elements.append({"type": "rich_text_preformatted", "elements": [{"type": "text", "text": f"\nResults:\n{df.to_string()}"}]})
        elif sql_execution_error:
            sql_blocks_elements.append({"type": "rich_text_preformatted", "elements": [{"type": "text", "text": f"\nError:\n{sql_execution_error}"}]})
        
        # --- AUTOMATIC FALLBACK LOGIC ---
        try:
            # First, TRY to send the results as a normal message
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

        except SlackApiError as e:
            if e.response["error"] == "msg_blocks_too_long":
                print("Message too long. Handling with fallback...")

                # --- NEW LOGIC STARTS HERE ---
                file_comment = "" # Initialize the comment for the file upload

                # First, check if the user requested the SQL
                if show_sql:
                    # Send a separate, smaller message with JUST the SQL query
                    say(
                        text=f"SQL for your query: {prompt}",
                        blocks=[
                            {
                                "type": "section",
                                "text": { "type": "mrkdwn", "text": f"The results for your query were too long to display, but here is the SQL you requested:" }
                            },
                            {
                                "type": "rich_text",
                                "elements": [{
                                    "type": "rich_text_preformatted",
                                    "elements": [{"type": "text", "text": sql}]
                                }]
                            }
                        ]
                    )
                    # Prepare a follow-up comment for the file
                    file_comment = "Here are the full results in the attached CSV file."
                else:
                    # If /sql was not used, the comment should be more descriptive
                    file_comment = f"The results for your query, '{prompt}', were too long to display directly. Here they are in the attached file."
                
                # --- NEW LOGIC ENDS HERE ---

                # Now, call the helper function to upload the file with the appropriate comment
                send_dataframe_as_csv(df, channel_id, file_comment, prompt)
            else:
                print(f"An unexpected Slack API error occurred: {e}")
                say(text=f"Sorry, I couldn't send the response due to a Slack error: `{e.response['error']}`")
    
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

def plot_chart(df):
    plt.figure(figsize=(10, 6), facecolor='#333333')

    # plot pie chart with percentages, using dynamic column names
    plt.pie(df[df.columns[1]], 
            labels=df[df.columns[0]], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=['#1f77b4', '#ff7f0e'], 
            textprops={'color':"white",'fontsize': 16})

    # ensure equal aspect ratio
    plt.axis('equal')
    # set the background color for the plot area to dark as well
    plt.gca().set_facecolor('#333333')   
    plt.tight_layout()

    # save the chart as a .jpg file
    file_path_jpg = 'pie_chart.jpg'
    plt.savefig(file_path_jpg, format='jpg')
    file_size = os.path.getsize(file_path_jpg)

    # upload image file to slack
    file_upload_url_response = app.client.files_getUploadURLExternal(filename=file_path_jpg,length=file_size)
    if DEBUG:
        print(file_upload_url_response)
    file_upload_url = file_upload_url_response['upload_url']
    file_id = file_upload_url_response['file_id']
    with open(file_path_jpg, 'rb') as f:
        response = requests.post(file_upload_url, files={'file': f})

    # check the response
    img_url = None
    if response.status_code != 200:
        print("File upload failed", response.text)
    else:
        # complete upload and get permalink to display
        response = app.client.files_completeUploadExternal(files=[{"id":file_id, "title":"chart"}])
        if DEBUG:
            print(response)
        img_url = response['files'][0]['permalink']
        time.sleep(2)
    
    return img_url

def init():
    conn,jwt,cortex_app = None,None,None

    conn = snowflake.connector.connect(
        user=USER,
        authenticator="SNOWFLAKE_JWT",
        private_key_file=RSA_PRIVATE_KEY_PATH,
        account=ACCOUNT,
        warehouse=WAREHOUSE,
        role=ROLE,
        host=HOST
    )
    if not conn.rest.token:
        print(">>>>>>>>>> Snowflake connection unsuccessful!")

    cortex_app = cortex_chat.CortexChat(
        AGENT_ENDPOINT, 
        #SEARCH_SERVICE,
        SEMANTIC_MODEL,
        MODEL, 
        ACCOUNT,
        USER,
        RSA_PRIVATE_KEY_PATH)

    print(">>>>>>>>>> Init complete")
    return conn,jwt,cortex_app

# Start app
if __name__ == "__main__":
    CONN,JWT,CORTEX_APP = init()
    Root = Root(CONN)
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
