import requests
import json
# import generate_jwt # Assuming generate_jwt.py is in the same directory or accessible in PYTHONPATH
from generate_jwt import JWTGenerator
import logging # Import logging

DEBUG = True 

class CortexChat:
    def __init__(self,
            agent_url: str, 
            semantic_model: str,
            model: str, 
            # We no longer need to pass the token or path here
            logger=None
        ):
        # Logger setup remains the same
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

        # The full URL to the Cortex Agent API
        self.agent_url = agent_url
        self.model = model
        self.semantic_model = semantic_model
        
    def _get_login_token(self) -> str:
        """
        Reads the fresh SPCS OAuth token from the file just before use.
        """
        token_path = "/snowflake/session/token"
        try:
            with open(token_path, "r") as f:
                return f.read()
        except Exception as e:
            self.logger.critical(f"FATAL: Could not read SPCS token file at {token_path}. Error: {e}", exc_info=True)
            # This is a critical failure, so we re-raise the exception.
            raise IOError(f"Could not read SPCS token from {token_path}") from e
        
    def _retrieve_response(self, query: str) -> dict[str, any]:
        self.logger.info("Preparing to send new request to Cortex Agent API.")
        
        fresh_token = self._get_login_token()

        headers = {
            "Authorization": f"Bearer {fresh_token}",
            "X-Snowflake-Authorization-Token-Type": "OAUTH",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        ## Local
        data = {
            "model": self.model,
            "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
            ],
            "tools": [
                {
                    "tool_spec": {
                        "type": "cortex_analyst_text_to_sql",
                        "name": "sql_analyst_tool"
                    }
                }
            ],
            "tool_resources": {
                "sql_analyst_tool": {
                    "semantic_model_file": self.semantic_model
                }
            },
        }
        response = requests.post(self.agent_url, headers=headers, json=data)

        self.logger.debug(f"--- Raw API Response Status Code: {response.status_code} ---")
        response_text_to_log = response.text[:1000] + ("..." if len(response.text) > 1000 else "")
        self.logger.debug(f"--- Raw API Response Text: {response_text_to_log} ---")

        if response.status_code == 200:
            return self._parse_response(response)
        else:
            self.logger.error(f"Cortex Agent API Error: Status {response.status_code}, Message: {response.text}")
            return None

    def _parse_delta_content(self, content_list: list) -> dict[str, any]: # Renamed content to content_list
        result = {'text': '', 'tool_use': [], 'tool_results': []}
        for entry in content_list:
            entry_type = entry.get('type')
            if entry_type == 'text': result['text'] += entry.get('text', '')
            elif entry_type == 'tool_use': result['tool_use'].append(entry.get('tool_use', {}))
            elif entry_type == 'tool_results': result['tool_results'].append(entry.get('tool_results', {}))
        return result

    def _process_sse_line(self, line: str) -> dict[str, any]:
        if not line.startswith('data: '): return {}
        try:
            json_str = line[6:].strip()
            if json_str == '[DONE]': return {'type': 'done'}
            data = json.loads(json_str)
            if data.get("code") and data.get("message"):
                 self.logger.error(f"SSE Error in stream: Code {data['code']} - {data['message']} (request_id: {data.get('request_id', 'N/A')})")
                 return {'type': 'error', 'data': data}
            if data.get('object') == 'message.delta':
                delta = data.get('delta', {})
                if 'content' in delta:
                    return {'type': 'message', 'content': self._parse_delta_content(delta['content'])}
            return {'type': 'other', 'data': data}
        except json.JSONDecodeError:
             self.logger.warning(f"Failed to parse SSE line as JSON: {line}")
             return {'type': 'non_json', 'raw': line}
        except Exception as e:
            self.logger.error(f"Error processing SSE line: {e} - Line: {line}", exc_info=True)
            return {'type': 'error', 'message': f'Processing error: {e}'}

    def _parse_response(self, response: requests.Response) -> dict[str, any]:
        accumulated = {'text': '', 'tool_use': [], 'tool_results': [], 'other': [], 'errors': []}
        self.logger.debug("Starting to parse SSE response stream...")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # self.logger.debug(f"SSE Line: {decoded_line}") # Very verbose, enable if desperate
                result = self._process_sse_line(decoded_line)
                if result.get('type') == 'message':
                    content_data = result['content']
                    accumulated['text'] += content_data['text']
                    accumulated['tool_use'].extend(content_data['tool_use'])
                    accumulated['tool_results'].extend(content_data['tool_results'])
                elif result.get('type') == 'other': accumulated['other'].append(result['data'])
                elif result.get('type') == 'error': accumulated['errors'].append(result.get('data') or result.get('message'))

        final_text = accumulated.get('text', '')
        sql = ''
        suggestions_list = []
        text_accompanying_suggestions = ''

        self.logger.debug(f"\n=== Accumulated Data (in _parse_response of CortexChat) ===")
        self.logger.debug(f"Accumulated Text: {accumulated['text']}")
        self.logger.debug(f"Accumulated Tool Use: {json.dumps(accumulated['tool_use'], indent=2)}")
        self.logger.debug(f"Accumulated Tool Results: {json.dumps(accumulated['tool_results'], indent=2)}")
        self.logger.debug(f"Accumulated Errors in Stream: {json.dumps(accumulated['errors'], indent=2)}")
        self.logger.debug("--- End Accumulated Data ---")

        if accumulated['tool_results']:
            for tool_result_item in accumulated['tool_results']:
                if 'content' in tool_result_item and isinstance(tool_result_item['content'], list):
                    for content_block in tool_result_item['content']:
                        if isinstance(content_block, dict) and 'json' in content_block and \
                           isinstance(content_block['json'], dict):
                            if 'sql' in content_block['json']:
                                sql = content_block['json']['sql']
                                self.logger.info(f"Extracted SQL: {sql}")
                            if 'suggestions' in content_block['json'] and \
                               isinstance(content_block['json']['suggestions'], list):
                                suggestions_list.extend(content_block['json']['suggestions'])
                                self.logger.info(f"Extracted suggestions: {content_block['json']['suggestions']}")
                                if 'text' in content_block['json']:
                                    text_accompanying_suggestions = content_block['json']['text']
                                    self.logger.info(f"Extracted text accompanying suggestions: {text_accompanying_suggestions}")

        if text_accompanying_suggestions:
            final_text = text_accompanying_suggestions
        elif not sql and not suggestions_list and not final_text:
            if accumulated['tool_use'] and not accumulated['errors']:
                 final_text = "I used my tools but didn't find a specific answer or SQL query."
                 self.logger.info("Setting default text as tools were used but no specific output parsed.")

        if accumulated['errors']:
            self.logger.error(f"Errors detected during SSE response parsing: {accumulated['errors']}")
            # Consider if final_text should indicate an error here
            # final_text += " (An error occurred during response processing)"


        self.logger.info(f"Returning from _parse_response. Final Text: '{final_text[:100]}...', SQL: '{sql[:100]}...', Suggestions Count: {len(suggestions_list)}")
        return {"text": final_text, "sql": sql, "suggestions": suggestions_list}

    def chat(self, query: str) -> dict[str, any]:
        self.logger.info(f"Received chat query: {query}")
        response_data = self._retrieve_response(query)
        if response_data is None:
            self.logger.warning("chat: _retrieve_response returned None. Returning error message dict.")
            return {"text": "Error: Failed to retrieve a response from the Cortex Agent API.", "sql": "", "suggestions": []}
        self.logger.info("chat: Successfully retrieved and parsed response.")
        return response_data