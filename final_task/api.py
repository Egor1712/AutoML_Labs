from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Загрузка
model = joblib.load("model.pkl")  # Исправлено: модель находится в /app/model.pkl

# Определение структуры данных для запроса
class PredictionRequest(BaseModel):
    feature1: float

# Определение структуры данных для ответа
class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    data = [[request.feature1]]
    prediction = model.predict(data)
    return PredictionResponse(prediction=prediction[0])  # Исправлен возврат

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)