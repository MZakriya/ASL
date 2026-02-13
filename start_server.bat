@echo off
echo Starting Sign Language Recognition API...
echo Make sure you have installed the required dependencies with: pip install -r requirements.txt
echo.
echo Please ensure the following files are in the project directory:
echo - sign_language_FINAL_A100_SUCCESS.pth (your trained model)
echo - vocab.pkl (your vocabulary file)
echo.
echo Starting server on http://localhost:8000
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000