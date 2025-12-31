import boto3
import json

# Create a Bedrock Runtime client in the AWS Region.
client = boto3.client("bedrock-runtime")

# Set the model id - Inference Profile
model_id = "meta.llama3-1-70b-instruct-v1:0"

# Define prompt
prompt = "Describe the purpose of a 'hello world' program in one line."

# Embed the prompt in Llama 3's instruction format.
formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Format the request payload using the model's native structure.
request_native = {
    "prompt": formatted_prompt,
    "max_gen_length": 512,
    "temperature": 0.5,
}

# Convert the native request to JSON.
request = json.dumps(request_native)

# Invoke the model with the request.
response = client.invoke_model(modelId= model_id , body=request)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract and print the response text.
response_text = model_response["generation"]
print(response_text)