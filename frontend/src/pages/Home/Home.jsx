import { useState } from 'react'
import TrainingView from '../Training/TrainingView'
import ResultsView from '../Training/ResultsView'
import '../../styles/Home.css'

function Home() {
  const [showTraining, setShowTraining] = useState(false)
  const [trainingResults, setTrainingResults] = useState([]) // Array to store all training runs
  const [selectedDataset, setSelectedDataset] = useState("cats_dogs") // Default to cats_dogs

  const handleTrainingComplete = (newResults) => {
    // Add timestamp to results for identification
    const resultsWithTimestamp = {
      ...newResults,
      timestamp: new Date().toISOString(),
      runNumber: trainingResults.length + 1
    }
    setTrainingResults([...trainingResults, resultsWithTimestamp])
    setShowTraining(false) // Hide training view to show results
  }

  const handleClearResults = () => {
    setTrainingResults([])
  }

  if (showTraining) {
    return (
      <TrainingView 
        onBack={() => setShowTraining(false)}
        onTrainingComplete={handleTrainingComplete}
        datasetType={selectedDataset}
      />
    )
  }

  if (trainingResults.length > 0) {
    return (
      <ResultsView 
        results={trainingResults}
        onRetrain={() => {
          setShowTraining(true)
        }}
        onBack={() => {
          setShowTraining(false)
          setTrainingResults([])
        }}
        onClearResults={handleClearResults}
      />
    )
  }

  return (
    <div className="app">
      <div className="container">
        <h1>AI Model Training App</h1>
        <p className="subtitle">Select a dataset to begin training your logistic regression model.</p>

        <div className="dataset-selection">
          <div
            className={`dataset-card ${selectedDataset === "cats_dogs" ? "selected" : ""}`}
            onClick={() => setSelectedDataset("cats_dogs")}
          >
            <h2>Cats vs Dogs</h2>
            <p>Binary classification: Cat or Dog</p>
          </div>
          <div
            className={`dataset-card ${selectedDataset === "brain_tumor" ? "selected" : ""}`}
            onClick={() => setSelectedDataset("brain_tumor")}
          >
            <h2>Brain Tumor Detection</h2>
            <p>Binary classification: Tumor or No Tumor (MRI images)</p>
          </div>
        </div>
        
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

