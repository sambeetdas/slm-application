from fastapi import FastAPI
from services.slm_service import SlmService
from models.ask_model import AskModel

app = FastAPI()

@app.post("/ask")
async def ask(request: AskModel):
    obj = SlmService()
    return obj.process_question(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)