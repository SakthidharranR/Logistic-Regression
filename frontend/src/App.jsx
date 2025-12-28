import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [number, setNumber] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const numValue = parseFloat(number)
      if (isNaN(numValue)) {
        setError('Please enter a valid number')
        setLoading(false)
        return
      }

      const response = await axios.post('/api/calculate', {
        number: numValue
      })

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred. Make sure the backend is running.')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="container">
        <h1>AI Calculation App</h1>
        <p className="subtitle">Enter a number to square it using NumPy</p>
        
        <form onSubmit={handleSubmit} className="form">
          <div className="input-group">
            <label htmlFor="number">Enter a number:</label>
            <input
              id="number"
              type="number"
              step="any"
              value={number}
              onChange={(e) => setNumber(e.target.value)}
              placeholder="e.g., 5"
              required
              disabled={loading}
            />
          </div>
          
          <button type="submit" disabled={loading} className="submit-btn">
            {loading ? 'Calculating...' : 'Calculate Square'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {result && (
          <div className="result">
            <h2>Result</h2>
            <div className="result-content">
              <p className="result-line">
                <span className="label">Input:</span>
                <span className="value">{result.input}</span>
              </p>
              <p className="result-line">
                <span className="label">Squared:</span>
                <span className="value highlight">{result.result}</span>
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

