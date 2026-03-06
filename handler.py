import runpod

def handler(event):
    input_data = event["input"]

    return {
        "message": "Stable Hair API running",
        "input": input_data
    }

runpod.serverless.start({
    "handler": handler
})
