"This module provides the CortexChat class to interact with the Snowflake Cortex Agent API."

import requests
import json
import logging
import sys


class CortexChat:
    """
    A class to interact with the Snowflake Cortex Agent API.

    This class handles the communication with the Cortex Agent API, including
    authentication, sending requests, and parsing the SSE (Server-Sent Events)
    stream responses.
    """

    def __init__(self, agent_url: str, semantic_model: str, model: str, logger=None):
        """
        Initializes the CortexChat instance.

        Args:
            agent_url: The URL of the Cortex Agent API.
            semantic_model: The semantic model to be used by the agent.
            model: The model to be used by the agent.
            logger: An optional logger instance. If not provided, a new logger
                    will be created.
        """
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

        self.agent_url = agent_url
        self.model = model
        self.semantic_model = semantic_model

    def _get_login_token(self) -> str:
        """
        Reads the fresh SPCS OAuth token from the file just before use.

        Returns:
            The SPCS OAuth token as a string.

        Raises:
            IOError: If the token file cannot be read.
        """
        token_path = "/snowflake/session/token"
        try:
            with open(token_path, "r") as f:
                return f.read()
        except Exception as e:
            self.logger.critical(
                f"FATAL: Could not read SPCS token file at {token_path}. Error: {e}",
                exc_info=True,
            )
            raise IOError(f"Could not read SPCS token from {token_path}") from e

    def _retrieve_response(self, query: str) -> dict[str, any]:
        """
        Sends a query to the Cortex Agent API and retrieves the response.

        Args:
            query: The user's query to send to the agent.

        Returns:
            A dictionary containing the parsed response from the agent, or None
            if an error occurred.
        """
        self.logger.info("Preparing to send new request to Cortex Agent API.")

        fresh_token = self._get_login_token()

        headers = {
            "Authorization": f"Bearer {fresh_token}",
            "X-Snowflake-Authorization-Token-Type": "OAUTH",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": query}]}
            ],
            "tools": [
                {
                    "tool_spec": {
                        "type": "cortex_analyst_text_to_sql",
                        "name": "sql_analyst_tool",
                    }
                }
            ],
            "tool_resources": {
                "sql_analyst_tool": {"semantic_model_file": self.semantic_model}
            },
        }
        response = requests.post(self.agent_url, headers=headers, json=data)

        # request_id = response.headers.get("X-Snowflake-Request-Id", "Not Found")
        # self.logger.debug(f"---- Agent Request ID: {request_id} -------ի")

        if response.status_code == 200:
            return self._parse_response(response)
        else:
            self.logger.error(
                f"Cortex Agent API Error: Status {response.status_code}, Message: {response.text}"
            )
            return None

    def _parse_delta_content(self, content_list: list) -> dict[str, any]:
        """
        Parses the content of a delta message from the SSE stream.

        Args:
            content_list: A list of content blocks from a delta message.

        Returns:
            A dictionary containing the parsed content, separated by type
            (text, tool_use, tool_results).
        """
        result = {"text": "", "tool_use": [], "tool_results": []}
        for entry in content_list:
            entry_type = entry.get("type")
            if entry_type == "text":
                result["text"] += entry.get("text", "")
            elif entry_type == "tool_use":
                result["tool_use"].append(entry.get("tool_use", {}))
            elif entry_type == "tool_results":
                result["tool_results"].append(entry.get("tool_results", {}))
        return result

    def _process_sse_line(self, line: str) -> dict[str, any]:
        """
        Processes a single line from the SSE stream.

        Args:
            line: A single line from the SSE stream.

        Returns:
            A dictionary representing the processed line.
        """
        if not line.startswith("data: "):
            return {}
        try:
            json_str = line[6:].strip()
            if json_str == "[DONE]":
                return {"type": "done"}
            data = json.loads(json_str)
            if data.get("code") and data.get("message"):
                self.logger.error(
                    f"SSE Error in stream: Code {data['code']} - {data['message']} (request_id: {data.get('request_id', 'N/A')})"
                )
                return {"type": "error", "data": data}
            if data.get("object") == "message.delta":
                delta = data.get("delta", {})
                if "content" in delta:
                    return {
                        "type": "message",
                        "content": self._parse_delta_content(delta["content"]),
                    }
            return {"type": "other", "data": data}
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse SSE line as JSON: {line}")
            return {"type": "non_json", "raw": line}
        except Exception as e:
            self.logger.error(
                f"Error processing SSE line: {e} - Line: {line}", exc_info=True
            )
            return {"type": "error", "message": f"Processing error: {e}"}

    def _parse_response(self, response: requests.Response) -> dict[str, any]:
        """
        Parses the entire SSE response stream from the Cortex Agent API.

        Args:
            response: The HTTP response object from the API.

        Returns:
            A dictionary containing the final aggregated response, including
            text, SQL, and suggestions.
        """
        accumulated = {
            "text": "",
            "tool_use": [],
            "tool_results": [],
            "other": [],
            "errors": [],
        }
        self.logger.debug("Starting to parse SSE response stream...")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                # The following line is useful for debugging SSE streams, but is very verbose.
                result = self._process_sse_line(decoded_line)
                if result.get("type") == "message":
                    content_data = result["content"]
                    accumulated["text"] += content_data["text"]
                    accumulated["tool_use"].extend(content_data["tool_use"])
                    accumulated["tool_results"].extend(content_data["tool_results"])
                elif result.get("type") == "other":
                    accumulated["other"].append(result["data"])
                elif result.get("type") == "error":
                    accumulated["errors"].append(
                        result.get("data") or result.get("message")
                    )

        final_text = accumulated.get("text", "")
        sql = ""
        suggestions_list = []
        text_accompanying_suggestions = ""

        if accumulated["tool_results"]:
            for tool_result_item in accumulated["tool_results"]:
                if "content" in tool_result_item and isinstance(
                    tool_result_item["content"], list
                ):
                    for content_block in tool_result_item["content"]:
                        if (
                            isinstance(content_block, dict)
                            and "json" in content_block
                            and isinstance(content_block["json"], dict)
                        ):
                            if "sql" in content_block["json"]:
                                sql = content_block["json"]["sql"]
                                self.logger.info(f"Extracted SQL: {sql}")
                            if "suggestions" in content_block["json"] and isinstance(
                                content_block["json"]["suggestions"], list
                            ):
                                suggestions_list.extend(
                                    content_block["json"]["suggestions"]
                                )
                                self.logger.info(
                                    f"Extracted suggestions: {content_block['json']['suggestions']}"
                                )
                                if "text" in content_block["json"]:
                                    text_accompanying_suggestions = content_block[
                                        "json"
                                    ]["text"]
                                    self.logger.info(
                                        f"Extracted text accompanying suggestions: {text_accompanying_suggestions}"
                                    )

        if text_accompanying_suggestions:
            final_text = text_accompanying_suggestions
        elif not sql and not suggestions_list and not final_text:
            if accumulated["tool_use"] and not accumulated["errors"]:
                final_text = (
                    "I used my tools but didn't find a specific answer or SQL query."
                )
                self.logger.info(
                    "Setting default text as tools were used but no specific output parsed."
                )

        if accumulated["errors"]:
            self.logger.error(
                f"Errors detected during SSE response parsing: {accumulated['errors']}"
            )

        if final_text.strip().lower().startswith("i apologize"):
            self.logger.info(
                "Detected 'I apologize' response. Replacing with custom message and link."
            )
            streamlit_app_url = "https://app.snowflake.com/rea28857/fanatics_collectibles_prod/#/streamlit-apps/FL_DATA_PROD.ML_DM.Q7WKFMFD8TSJT4YG?ref=snowsight_shared"
            final_text = f"I apologize, but I cannot provide a complete response, please visit the <{streamlit_app_url}|Fanatics analyst> on snowflake and continue there."
            sql = ""
            suggestions_list = []

        self.logger.info(
            f"Returning from _parse_response. Final Text: '{final_text[:100]}...', SQL: '{sql[:100]}...', Suggestions Count: {len(suggestions_list)}"
        )
        return {"text": final_text, "sql": sql, "suggestions": suggestions_list}

    def chat(self, query: str) -> dict[str, any]:
        """
        The main method to interact with the Cortex Agent.

        Args:
            query: The user's query.

        Returns:
            A dictionary containing the agent's response.
        """
        self.logger.info(f"Received chat query: {query}")
        response_data = self._retrieve_response(query)
        if response_data is None:
            self.logger.warning(
                "chat: _retrieve_response returned None. Returning error message dict."
            )
            return {
                "text": "Error: Failed to retrieve a response from the Cortex Agent API.",
                "sql": "",
                "suggestions": [],
            }
        self.logger.info("chat: Successfully retrieved and parsed response.")
        return response_data
