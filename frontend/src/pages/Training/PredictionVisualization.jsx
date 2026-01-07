import { useState, useEffect } from 'react'
import axios from 'axios'
import '../../styles/PredictionVisualization.css'

const API_BASE_URL = 'http://localhost:8000'
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000
})

function PredictionVisualization({ result, datasetType, onBack }) {
  const [testImages, setTestImages] = useState({})
  const [trainImages, setTrainImages] = useState({})
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('test') // 'test' or 'train'
  const [showCorrect, setShowCorrect] = useState(true)
  const [showIncorrect, setShowIncorrect] = useState(true)

  const class0Label = datasetType === "brain_tumor" ? "No Tumor" : "Cat"
  const class1Label = datasetType === "brain_tumor" ? "Tumor" : "Dog"

  useEffect(() => {
    const loadImages = async () => {
      setLoading(true)
      const imagesToLoad = {}
      
      // Load test set images
      const testStartIndex = result.test_start_index || (result.total_available - result.test_set_size)
      if (result.test_predictions) {
        for (const pred of result.test_predictions) {
          const actualIndex = testStartIndex + pred.relative_index
          try {
            const response = await api.get(`/api/images/preview/${actualIndex}`, {
              params: { size: 128, dataset_type: datasetType }
            })
            imagesToLoad[`test_${pred.relative_index}`] = response.data.image
          } catch (err) {
            console.error(`Error loading test image ${actualIndex}:`, err)
          }
        }
        setTestImages(imagesToLoad)
      }
      
      // Load training set images
      const trainStartIndex = result.training_start_index || 0
      const trainImagesToLoad = {}
      if (result.train_predictions) {
        for (const pred of result.train_predictions) {
          const actualIndex = trainStartIndex + pred.relative_index
          try {
            const response = await api.get(`/api/images/preview/${actualIndex}`, {
              params: { size: 128, dataset_type: datasetType }
            })
            trainImagesToLoad[`train_${pred.relative_index}`] = response.data.image
          } catch (err) {
            console.error(`Error loading train image ${actualIndex}:`, err)
          }
        }
        setTrainImages(trainImagesToLoad)
      }
      
      setLoading(false)
    }

    if (result) {
      loadImages()
    }
  }, [result, datasetType])

  const getPredictions = () => {
    return activeTab === 'test' ? (result.test_predictions || []) : (result.train_predictions || [])
  }

  const getImages = () => {
    return activeTab === 'test' ? testImages : trainImages
  }

  const getCorrectCount = () => {
    return activeTab === 'test' ? (result.test_correct_count || 0) : (result.train_correct_count || 0)
  }

  const getIncorrectCount = () => {
    return activeTab === 'test' ? (result.test_incorrect_count || 0) : (result.train_incorrect_count || 0)
  }

  const predictions = getPredictions()
  const images = getImages()
  const correctPredictions = predictions.filter(p => p.correct && showCorrect)
  const incorrectPredictions = predictions.filter(p => !p.correct && showIncorrect)

  const getLabelName = (label) => {
    return label === 0 ? class0Label : class1Label
  }

  if (loading) {
    return (
      <div className="prediction-visualization">
        <div className="container">
          <div className="loading">Loading images...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="prediction-visualization">
      <div className="container">
        <div className="visualization-header">
          <button onClick={onBack} className="back-btn">← Back</button>
          <h1>Prediction Visualization</h1>
        </div>

        <div className="tabs">
          <button 
            className={activeTab === 'test' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('test')}
          >
            Test Set ({result.test_set_size} images)
          </button>
          <button 
            className={activeTab === 'train' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('train')}
          >
            Training Set ({result.training_set_size} images)
          </button>
        </div>

        <div className="filter-controls">
          <label className="filter-checkbox">
            <input
              type="checkbox"
              checked={showCorrect}
              onChange={(e) => setShowCorrect(e.target.checked)}
            />
            Show Correct ({getCorrectCount()})
          </label>
          <label className="filter-checkbox">
            <input
              type="checkbox"
              checked={showIncorrect}
              onChange={(e) => setShowIncorrect(e.target.checked)}
            />
            Show Incorrect ({getIncorrectCount()})
          </label>
        </div>

        <div className="predictions-grid">
          {/* Correct Predictions */}
          {showCorrect && correctPredictions.length > 0 && (
            <div className="prediction-section">
              <h2 className="section-title correct">
                ✓ Correctly Classified ({correctPredictions.length})
              </h2>
              <div className="images-grid">
                {correctPredictions.map((pred) => {
                  const imageKey = `${activeTab}_${pred.relative_index}`
                  const image = images[imageKey]
                  return (
                    <div key={pred.relative_index} className="prediction-card correct">
                      {image ? (
                        <img src={image} alt={`Image ${pred.relative_index}`} className="prediction-image" />
                      ) : (
                        <div className="image-placeholder">Loading...</div>
                      )}
                      <div className="prediction-info">
                        <div className="prediction-label">
                          <span className="label-name">Actual: {getLabelName(pred.actual)}</span>
                          <span className="label-name">Predicted: {getLabelName(pred.predicted)}</span>
                        </div>
                        <div className="prediction-status correct">✓ Correct</div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Incorrect Predictions */}
          {showIncorrect && incorrectPredictions.length > 0 && (
            <div className="prediction-section">
              <h2 className="section-title incorrect">
                ✗ Incorrectly Classified ({incorrectPredictions.length})
              </h2>
              <div className="images-grid">
                {incorrectPredictions.map((pred) => {
                  const imageKey = `${activeTab}_${pred.relative_index}`
                  const image = images[imageKey]
                  return (
                    <div key={pred.relative_index} className="prediction-card incorrect">
                      {image ? (
                        <img src={image} alt={`Image ${pred.relative_index}`} className="prediction-image" />
                      ) : (
                        <div className="image-placeholder">Loading...</div>
                      )}
                      <div className="prediction-info">
                        <div className="prediction-label">
                          <span className="label-name">Actual: {getLabelName(pred.actual)}</span>
                          <span className="label-name">Predicted: {getLabelName(pred.predicted)}</span>
                        </div>
                        <div className="prediction-status incorrect">✗ Incorrect</div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {predictions.length === 0 && (
            <div className="no-predictions">
              <p>No predictions available for visualization.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default PredictionVisualization

