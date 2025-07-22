import requests
import json
# import generate_jwt
from generate_jwt import JWTGenerator

# Keep DEBUG=True for testing the changes
DEBUG = True

class CortexChat:
    def __init__(self,
            agent_url: str,
            semantic_model: str,
            model: str,
            account: str,
            user: str,
            private_key_path: str
        ):
        self.agent_url = agent_url
        self.model = model
        self.semantic_model = semantic_model
        self.account = account
        self.user = user
        self.private_key_path = private_key_path
        self.jwt = JWTGenerator(self.account, self.user, self.private_key_path).get_token()

    # Removed limit parameter as it was only used for search
    def _retrieve_response(self, query: str) -> dict[str, any]:
        url = self.agent_url
        headers = {
            'X-Snowflake-Authorization-Token-Type': 'KEYPAIR_JWT',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f"Bearer {self.jwt}"
        }
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
                { # Kept only the cortex_analyst tool spec
                    "tool_spec": {
                        "type": "cortex_analyst_text_to_sql",
                        # You can keep the name or make it more generic if you prefer
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
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 401:  # Unauthorized - likely expired JWT
            print("JWT has expired. Generating new JWT...")
            self.jwt = JWTGenerator(self.account, self.user, self.private_key_path).get_token()
            headers["Authorization"] = f"Bearer {self.jwt}"
            print("New JWT generated. Sending new request to Cortex Agents API. Please wait...")
            response = requests.post(url, headers=headers, json=data)

        if DEBUG:
            # Print the raw response text to see exactly what the API returns
            print("--- Raw API Response Text ---")
            print(response.text)
            print("--- End Raw API Response Text ---")

        if response.status_code == 200:
            return self._parse_response(response)
        else:
            # Try to print JSON error message if possible, otherwise text
            try:
                error_msg = response.json()
            except json.JSONDecodeError:
                error_msg = response.text
            print(f"Error: Received status code {response.status_code} with message {error_msg}")
            return None # Return None on error

    def _parse_delta_content(self,content: list) -> dict[str, any]:
        """Parse different types of content from the delta."""
        result = {
            'text': '',
            'tool_use': [],
            'tool_results': []
        }

        for entry in content:
            entry_type = entry.get('type')
            if entry_type == 'text':
                result['text'] += entry.get('text', '')
            elif entry_type == 'tool_use':
                result['tool_use'].append(entry.get('tool_use', {}))
            elif entry_type == 'tool_results':
                result['tool_results'].append(entry.get('tool_results', {}))

        return result

    def _process_sse_line(self,line: str) -> dict[str, any]:
        """Process a single SSE line and return parsed content."""
        if not line.startswith('data: '):
            return {}
        try:
            json_str = line[6:].strip()
            if json_str == '[DONE]':
                return {'type': 'done'}

            data = json.loads(json_str)

            # Check for error messages within the SSE stream
            if data.get("code") and data.get("message"):
                 print(f"SSE Error: Code {data['code']} - {data['message']} (request_id: {data.get('request_id', 'N/A')})")
                 return {'type': 'error', 'data': data} # Propagate error info

            if data.get('object') == 'message.delta':
                delta = data.get('delta', {})
                if 'content' in delta:
                    return {
                        'type': 'message',
                        'content': self._parse_delta_content(delta['content'])
                    }
            return {'type': 'other', 'data': data}
        except json.JSONDecodeError:
            # Don't treat non-JSON lines (like potential error messages before JSON) as errors here
             if DEBUG: print(f"Warning: Failed to parse line as JSON: {line}")
             return {'type': 'non_json', 'raw': line}
        except Exception as e:
            print(f"Error processing SSE line: {e} - Line: {line}")
            return {'type': 'error', 'message': f'Processing error: {e}'}

    def _parse_response(self, response: requests.Response) -> dict[str, any]:
        """Parse the SSE chat response, extracting text, SQL, and suggestions."""
        accumulated = {
            'text': '',
            'tool_use': [],
            'tool_results': [],
            'other': [],
            'errors': []
        }

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                result = self._process_sse_line(decoded_line)

                if result.get('type') == 'message':
                    content_data = result['content'] # Renamed 'content' to 'content_data' to avoid confusion
                    accumulated['text'] += content_data['text']
                    accumulated['tool_use'].extend(content_data['tool_use'])
                    accumulated['tool_results'].extend(content_data['tool_results'])
                elif result.get('type') == 'other':
                    accumulated['other'].append(result['data'])
                elif result.get('type') == 'error':
                    accumulated['errors'].append(result.get('data') or result.get('message'))

        # Initialize what we want to return
        final_text = accumulated.get('text', '') # General text from LLM
        sql = ''
        suggestions_list = [] # Initialize as an empty list
        text_accompanying_suggestions = ''

        if DEBUG:
            print("\n=== Parsed Accumulated Data (in _parse_response) ===")
            # ... (your existing debug prints for accumulated data) ...

        if accumulated['tool_results']:
            for tool_result_item in accumulated['tool_results']:
                if 'content' in tool_result_item and isinstance(tool_result_item['content'], list):
                    for content_block in tool_result_item['content']:
                        if isinstance(content_block, dict) and 'json' in content_block and \
                           isinstance(content_block['json'], dict):
                            
                            # Extract SQL
                            if 'sql' in content_block['json']:
                                sql = content_block['json']['sql']

                            # Extract Suggestions and their accompanying text
                            if 'suggestions' in content_block['json'] and \
                               isinstance(content_block['json']['suggestions'], list):
                                suggestions_list.extend(content_block['json']['suggestions'])
                                if 'text' in content_block['json']:
                                    text_accompanying_suggestions = content_block['json']['text']

        # Logic to decide the primary 'text' to return:
        # If suggestions are present and have specific accompanying text, that text might be more relevant.
        if text_accompanying_suggestions:
            final_text = text_accompanying_suggestions # Prioritize text that came with suggestions
        elif not sql and not suggestions_list and not final_text: # If everything is empty, but tool use happened
            if accumulated['tool_use'] and not accumulated['errors']:
                 final_text = "I used my tools but didn't find a specific answer or SQL query."


        if accumulated['errors']:
            print("\n--- ERRORS DETECTED DURING RESPONSE PARSING ---")
            # ... (error printing logic) ...
            # Potentially override final_text if there's a critical error
            # final_text = "An error occurred while processing the response."

        return {"text": final_text, "sql": sql, "suggestions": suggestions_list}

    def chat(self, query: str) -> dict[str, any]:
        # Added error handling for the case where _retrieve_response returns None
        response_data = self._retrieve_response(query)
        if response_data is None:
            # Return a dictionary indicating failure, matching the expected return type
            return {"text": "Failed to retrieve response from API", "sql": ""}
        return response_data