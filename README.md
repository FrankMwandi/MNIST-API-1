# MNIST API (PyTorch + FastAPI)

This project serves a trained PyTorch model as a REST API using FastAPI.

## Features
- Trained CNN on MNIST
- `/predict` endpoint for digit classification
- Upload a 28x28 grayscale image and get the digit (0–9)

## Run it
```bash
uvicorn app:app --reload
