import runpod

def handler(event):
    input_data = event["input"]

    return {
        "status": "Stable Hair server running",
        "input_received": input_data
    }

runpod.serverless.start({
    "handler": handler
})
