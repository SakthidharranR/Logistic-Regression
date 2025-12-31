import { useState } from 'react'
import TrainingView from '../Training/TrainingView'
import '../../styles/Home.css'

function Home() {
  const [showTraining, setShowTraining] = useState(false)

  if (showTraining) {
    return <TrainingView onBack={() => setShowTraining(false)} />
  }

  return (
    <div className="app">
      <div className="container">
        <h1>AI Cat Training App</h1>
        <p className="subtitle">Train your logistic regression model on cat images</p>
        
        <div className="main-actions">
          <button 
            onClick={() => setShowTraining(true)}
            className="begin-training-btn"
          >
            Begin Training!
          </button>
        </div>
      </div>
    </div>
  )
}

export default Home

