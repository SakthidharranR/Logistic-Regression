# AI Calculation App

A full-stack application with FastAPI backend (optimized for AI/ML) and React frontend. This app allows users to enter a number, square it using NumPy, and display the result.

## Project Structure

```
Logistic Regression/
├── backend/          # FastAPI Python backend
│   ├── main.py      # FastAPI server with calculation endpoint
│   └── requirements.txt
├── frontend/         # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
└── README.md
```

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ and npm installed

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   ```
   Note: On macOS, use `python3` instead of `python` if `python` is not available.

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

   The backend will be available at `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## Usage

1. Make sure both backend and frontend servers are running
2. Open your browser and navigate to `http://localhost:5173`
3. Enter a number in the input field
4. Click "Calculate Square" to see the result

## API Endpoints

### POST `/api/calculate`

Squares a number using NumPy.

**Request Body:**
```json
{
  "number": 5
}
```

**Response:**
```json
{
  "input": 5,
  "result": 25.0,
  "operation": "square"
}
```

## Technologies Used

- **Backend**: FastAPI, NumPy, Uvicorn
- **Frontend**: React, Vite, Axios
- **Styling**: CSS3 with modern gradients and animations

## Development

- Backend runs on port 8000
- Frontend runs on port 5173 (Vite default)
- CORS is configured to allow communication between frontend and backend
- Vite proxy is configured to forward `/api` requests to the backend

