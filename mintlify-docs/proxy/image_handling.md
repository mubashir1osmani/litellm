
# Image URL Handling 

![](/images/image_handling.png)

Some LLM API's don't support url's for images, but do support base-64 strings. 

For those, LiteLLM will:

1. Detect a URL being passed
2. Check if the LLM API supports a URL
3. Else, will download the base64 
4. Send the provider a base64 string. 


LiteLLM also caches this result, in-memory to reduce latency for subsequent calls. 

The limit for an in-memory cache is 1MB. 