from fastapi import FastAPI
from services.slm_service import SlmService
from models.ask_model import AskModel

# Initialize FastAPI app
app = FastAPI()

@app.post("/ask")
async def ask(request: AskModel):
    obj = SlmService()

    """Load the model when the app starts"""
    await obj.load_model()

    return await obj.generate_response(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)