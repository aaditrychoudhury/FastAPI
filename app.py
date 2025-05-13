from fastapi import FastAPI, Request
from subprocess import run, PIPE
import json

app = FastAPI()

@app.post("/predict")
async def predict(request: Request): 
    body = await request.json()
    input_json = json.dumps(body)
    # Run the R script
    result = run(["Rscript", "predict.R", input_json], stdout=PIPE, stderr=PIPE, text=True)

    # If the R script fails, log and return the error
    if result.returncode != 0:
        print("R script error:")
        print(result.stderr)
        return {"error": result.stderr.strip()}

    # Try to parse JSON from R output
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Invalid JSON returned by R:")
        print(result.stdout)
        return {"error": "Invalid JSON returned by R script."}
