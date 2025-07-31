import json
import asyncio
from typing import Optional, Dict, Any
try:
    import aiohttp
except ImportError:
    aiohttp = None
from ..core.tool import Tool, tool, param


@tool(name="api_client", description="Make HTTP requests to APIs (GET, POST, PUT, DELETE)")
class APIClientTool(Tool):
    """
    A tool for making HTTP API requests.
    Supports GET, POST, PUT, DELETE methods with headers, query parameters, and JSON body.
    """
    
    @param("url", type="string", description="API endpoint URL")
    @param("method", type="string", description="HTTP method", enum=["GET", "POST", "PUT", "DELETE", "PATCH"], default="GET", required=False)
    @param("headers", type="object", description="HTTP headers as key-value pairs", required=False)
    @param("params", type="object", description="Query parameters as key-value pairs", required=False)
    @param("json_body", type="object", description="JSON request body for POST/PUT/PATCH requests", required=False)
    @param("timeout", type="number", description="Request timeout in seconds", default=30, required=False)
    async def _execute(self) -> str:
        """Make an HTTP API request."""
        url = self.get_param("url")
        method = self.get_param("method", "GET").upper()
        headers = self.get_param("headers", {})
        params = self.get_param("params", {})
        json_body = self.get_param("json_body")
        timeout = self.get_param("timeout", 30)
        
        if not url:
            return "Error: URL parameter is required"
        
        if not aiohttp:
            return "Error: aiohttp package is required for API client. Install with: pip install aiohttp"
        
        # Validate method
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            return f"Error: Unsupported HTTP method '{method}'"
        
        # Set default headers
        if not headers:
            headers = {}
        if json_body and "content-type" not in [h.lower() for h in headers.keys()]:
            headers["Content-Type"] = "application/json"
        
        try:
            # Create timeout
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                # Prepare request parameters
                request_kwargs = {
                    "headers": headers,
                    "params": params
                }
                
                # Add JSON body for methods that support it
                if json_body and method in ["POST", "PUT", "PATCH"]:
                    request_kwargs["json"] = json_body
                
                self._log_info(f"Making {method} request to: {url}")
                
                # Make the request
                async with session.request(method, url, **request_kwargs) as response:
                    # Get response details
                    status_code = response.status
                    response_headers = dict(response.headers)
                    
                    # Try to get response body
                    try:
                        if response.content_type == 'application/json':
                            response_data = await response.json()
                        else:
                            response_data = await response.text()
                    except Exception as e:
                        response_data = f"Could not parse response body: {str(e)}"
                    
                    # Format response
                    result = {
                        "status_code": status_code,
                        "headers": response_headers,
                        "data": response_data,
                        "url": str(response.url),
                        "method": method
                    }
                    
                    # Log result
                    self._log_info(f"API response: {status_code} from {url}")
                    
                    # Return formatted response
                    if status_code >= 200 and status_code < 300:
                        return self._format_success_response(result)
                    else:
                        return self._format_error_response(result)
                        
        except asyncio.TimeoutError:
            error_msg = f"Request timed out after {timeout} seconds"
            self._log_error(f"API request timeout: {url}")
            return f"Error: {error_msg}"
        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error: {str(e)}"
            self._log_error(f"API client error: {str(e)}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error making API request: {str(e)}"
            self._log_error(f"API request failed: {str(e)}")
            return f"Error: {error_msg}"
    
    def _format_success_response(self, result: Dict[str, Any]) -> str:
        """Format a successful API response."""
        output = f"✅ API Request Successful\n"
        output += f"Status: {result['status_code']}\n"
        output += f"URL: {result['url']}\n"
        output += f"Method: {result['method']}\n"
        
        # Add response data
        if isinstance(result['data'], dict) or isinstance(result['data'], list):
            # Pretty print JSON
            try:
                data_str = json.dumps(result['data'], indent=2)
                if len(data_str) > 2000:  # Truncate very long responses
                    data_str = data_str[:2000] + "\n... (response truncated)"
                output += f"\nResponse:\n{data_str}"
            except Exception:
                output += f"\nResponse: {str(result['data'])[:1000]}"
        else:
            # Plain text response
            response_str = str(result['data'])
            if len(response_str) > 1000:
                response_str = response_str[:1000] + "... (response truncated)"
            output += f"\nResponse: {response_str}"
        
        return output
    
    def _format_error_response(self, result: Dict[str, Any]) -> str:
        """Format an error API response."""
        output = f"❌ API Request Failed\n"
        output += f"Status: {result['status_code']}\n"
        output += f"URL: {result['url']}\n"
        output += f"Method: {result['method']}\n"
        
        # Add error details
        if isinstance(result['data'], dict) or isinstance(result['data'], list):
            try:
                data_str = json.dumps(result['data'], indent=2)
                output += f"\nError Response:\n{data_str}"
            except Exception:
                output += f"\nError Response: {str(result['data'])}"
        else:
            output += f"\nError Response: {str(result['data'])}"
        
        return output