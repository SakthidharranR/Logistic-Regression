import { useState, useEffect } from 'react'
import axios from 'axios'
import '../../styles/TrainingView.css'

// Configure axios to use backend URL
const API_BASE_URL = 'http://localhost:8000'
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000  // 10 seconds for regular requests
})

// Separate axios instance for training with longer timeout (5 minutes)
const trainingApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000  // 5 minutes for training requests
})

function TrainingView({ onBack, onTrainingComplete, datasetType = "cats_dogs" }) {
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [numImages, setNumImages] = useState(null)
  const [numImagesInput, setNumImagesInput] = useState('')
  const [learningRate, setLearningRate] = useState(0.005)
  const [numIterations, setNumIterations] = useState(2000)
  const [numTest, setNumTest] = useState(1000)
  const [numTrainingCats, setNumTrainingCats] = useState(null)
  const [numTrainingDogs, setNumTrainingDogs] = useState(null)
  const [numTestCats, setNumTestCats] = useState(null)
  const [numTestDogs, setNumTestDogs] = useState(null)
  const [trainingStarted, setTrainingStarted] = useState(false)
  const [trainingInProgress, setTrainingInProgress] = useState(false)
  const [imagePreviews, setImagePreviews] = useState({})
  const [currentPage, setCurrentPage] = useState(0)
  const imagesPerPage = 10

  // Load image list on mount
  useEffect(() => {
    let cancelled = false
    
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        
        // Add timeout to prevent hanging
        const response = await api.get('/api/images/list')
        
        if (cancelled) return
        
        if (response.data.error) {
          setError(response.data.error)
          setImages([])
          setLoading(false)
          return
        }
        
        // If backend returns empty images array but has total_count, generate indices
        let imageList = response.data.images || []
        const totalCount = response.data.total_count || 0
        
        // If we have total_count but no images array, generate indices
        if (totalCount > 0 && imageList.length === 0) {
          const maxToGenerate = datasetType === "brain_tumor" ? totalCount : Math.min(11000, totalCount)
          setMaxImages(maxToGenerate)
          imageList = Array.from({ length: maxToGenerate }, (_, i) => i)
        } else if (totalCount > 0) {
          const maxToGenerate = datasetType === "brain_tumor" ? totalCount : Math.min(11000, totalCount)
          setMaxImages(maxToGenerate)
        } else {
          setMaxImages(0)
        }
        
        setImages(imageList)
        
        if (totalCount) {
          // Don't set initial value - let user enter it
          // Just ensure we have enough images available
        }
        
        // Clear loading state immediately after getting the list
        setLoading(false)
        
        // Load previews asynchronously (don't block on this)
        // Load first page of 10 images
        if (imageList.length > 0 && !cancelled) {
          const initialBatch = imageList.slice(0, imagesPerPage) // Load first 10 images
          // Don't await - let it load in background
          loadImagePreviews(initialBatch).catch(err => {
            console.error('Error loading previews:', err)
          })
        }
      } catch (err) {
        if (cancelled) return
        
        const errorMessage = err.code === 'ERR_NETWORK' || err.code === 'ECONNABORTED' || err.message?.includes('Network Error') || err.message?.includes('ERR_CONNECTION_REFUSED') || err.message?.includes('timeout')
          ? 'Cannot connect to backend server. Make sure the backend is running on http://localhost:8000'
          : err.response?.data?.detail || err.response?.data?.error || 'Failed to load images'
        
        setError(errorMessage)
        setImages([])
        setLoading(false)
        console.error('Error loading images:', err)
      }
    }
    
    fetchData()
    
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetType])

  // loadImages function removed - now handled directly in useEffect

  const loadImagePreviews = async (imageList) => {
    // TEST: Load only 1 image at a time with longer timeout and better logging
    console.log('Loading image previews:', imageList)
    for (const imageId of imageList) {
      try {
        // Convert imageId to string for URL
        const imageIdStr = String(imageId)
        console.log(`Loading image ${imageIdStr}...`)
        const startTime = Date.now()
        const response = await api.get(`/api/images/preview/${imageIdStr}`, {
          params: { size: 64, dataset_type: datasetType },
          timeout: 30000 // 30 second timeout for testing
        })
        const endTime = Date.now()
        console.log(`Successfully loaded image ${imageIdStr} in ${endTime - startTime}ms`)
        setImagePreviews(prev => ({ ...prev, [imageId]: response.data.image }))
      } catch (err) {
        console.error(`Error loading preview for ${imageId}:`, err)
        // Don't fail silently - log the error for debugging
      }
    }
  }

  const handleNumImagesChange = (e) => {
    const inputValue = e.target.value
    setNumImagesInput(inputValue) // Store raw input - allow free typing
    
    // Allow empty input for easier editing
    if (inputValue === '' || inputValue === '-') {
      setNumImages(null) // Clear the numeric value
      return
    }
    
    // Only update numImages if it's a valid number, but don't clamp during typing
    const value = parseInt(inputValue)
    if (!isNaN(value)) {
      setNumImages(value) // Store the value but don't clamp the input
    } else {
      setNumImages(null) // Invalid input
    }
  }
  
  const handleNumImagesBlur = () => {
    // When input loses focus, validate and clamp if needed
    if (numImagesInput === '' || isNaN(parseInt(numImagesInput))) {
      setNumImages(null)
      // Keep it empty - don't reset to default
      return
    }
    
    const value = parseInt(numImagesInput)
    // Clamp to valid range (1-11000) only on blur
    if (value < 1) {
      setNumImages(1)
      setNumImagesInput('1')
    } else if (value > 11000) {
      setNumImages(11000)
      setNumImagesInput('11000')
    } else {
      setNumImages(value)
      setNumImagesInput(String(value))
    }
  }

  const handleStartTraining = async () => {
    // Get the actual numeric value
    const actualNumImages = typeof numImages === 'number' ? numImages : (numImagesInput ? parseInt(numImagesInput) : null)
    
    if (!actualNumImages || actualNumImages < 1 || actualNumImages > maxImages) {
      setError(`Number of images must be between 1 and ${maxImages}`)
      return
    }

    if (actualNumImages + numTest > images.length) {
      setError(`Not enough images. Available: ${images.length}, Need: ${actualNumImages} (training) + ${numTest} (testing) = ${actualNumImages + numTest}`)
      return
    }

    if (learningRate <= 0 || learningRate > 1) {
      setError('Learning rate must be between 0 and 1')
      return
    }

    if (numIterations < 100 || numIterations > 10000) {
      setError('Number of iterations must be between 100 and 10,000')
      return
    }

    if (numTest < 100 || numTest > 5000) {
      setError('Number of test images must be between 100 and 5,000')
      return
    }

    // Validate cat/dog counts if provided
    if (numTrainingCats !== null || numTrainingDogs !== null) {
      const trainingCats = numTrainingCats || 0
      const trainingDogs = numTrainingDogs || 0
      if (trainingCats + trainingDogs !== actualNumImages) {
        setError(`Training cats (${trainingCats}) + dogs (${trainingDogs}) must equal training images (${actualNumImages})`)
        return
      }
      if (trainingCats < 0 || trainingDogs < 0) {
        setError('Cat and dog counts cannot be negative')
        return
      }
    }

    if (numTestCats !== null || numTestDogs !== null) {
      const testCats = numTestCats || 0
      const testDogs = numTestDogs || 0
      if (testCats + testDogs !== numTest) {
        setError(`Test cats (${testCats}) + dogs (${testDogs}) must equal test images (${numTest})`)
        return
      }
      if (testCats < 0 || testDogs < 0) {
        setError('Cat and dog counts cannot be negative')
        return
      }
    }

    try {
      setError(null)
      setTrainingInProgress(true)  // Show training progress indicator
      setTrainingStarted(false)
      const response = await trainingApi.post('/api/training/start', {
        num_images: actualNumImages,
        learning_rate: learningRate,
        num_iterations: numIterations,
        num_test: numTest,
        dataset_type: datasetType,
        num_training_cats: datasetType === "cats_dogs" ? numTrainingCats : null,
        num_training_dogs: datasetType === "cats_dogs" ? numTrainingDogs : null,
        num_training_no_tumor: datasetType === "brain_tumor" ? numTrainingCats : null,
        num_training_tumor: datasetType === "brain_tumor" ? numTrainingDogs : null,
        num_test_cats: datasetType === "cats_dogs" ? numTestCats : null,
        num_test_dogs: datasetType === "cats_dogs" ? numTestDogs : null,
        num_test_no_tumor: datasetType === "brain_tumor" ? numTestCats : null,
        num_test_tumor: datasetType === "brain_tumor" ? numTestDogs : null,
      })
      setTrainingInProgress(false)
      console.log('Training completed:', response.data)
      
      // Log debug information if available
      if (response.data.debug_logs) {
        console.log('\n=== BACKEND DEBUG LOGS ===')
        response.data.debug_logs.forEach(log => console.log(log))
        console.log('===========================\n')
      }

      // Navigate to results page
      if (onTrainingComplete) {
        onTrainingComplete(response.data)
      } else {
        setTrainingStarted(true)
      }
    } catch (err) {
      setTrainingInProgress(false)
      const errorMessage = err.response?.data?.detail || err.response?.data?.error || err.message || 'Failed to start training'
      setError(errorMessage)
      console.error('Error starting training:', err)
      console.error('Error response:', err.response?.data)
    }
  }

  // Calculate pagination - use actual numeric value (default to 0 if empty to show nothing)
  const actualNumImages = typeof numImages === 'number' ? numImages : (numImagesInput ? parseInt(numImagesInput) : 0)
  const validNumImages = actualNumImages > 0 ? actualNumImages : 0
  const totalPages = validNumImages > 0 ? Math.ceil(Math.min(validNumImages, images.length) / imagesPerPage) : 0
  const startIndex = currentPage * imagesPerPage
  const endIndex = validNumImages > 0 ? Math.min(startIndex + imagesPerPage, Math.min(validNumImages, images.length)) : 0
  const displayedImages = validNumImages > 0 ? images.slice(startIndex, endIndex) : []
  
  // Load images for current page when page changes
  useEffect(() => {
    if (displayedImages.length > 0) {
      const imagesToLoad = displayedImages.filter(img => !imagePreviews[img] && !imagePreviews[String(img)])
      if (imagesToLoad.length > 0) {
        console.log(`Loading page ${currentPage + 1}: images ${startIndex} to ${endIndex - 1}`)
        loadImagePreviews(imagesToLoad)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage, numImages, numImagesInput, images.length])

  const handlePreviousPage = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1)
    }
  }

  const handleNextPage = () => {
    if (currentPage < totalPages - 1) {
      setCurrentPage(currentPage + 1)
    }
  }

  // Reset to page 0 when numImages changes
  useEffect(() => {
    setCurrentPage(0)
  }, [numImages, numImagesInput])

  if (loading && images.length === 0) {
    return (
      <div className="training-view">
        <div className="container">
          <div className="loading">Loading images...</div>
          {error && (
            <div className="error-message" style={{marginTop: '20px'}}>
              {error}
            </div>
          )}
        </div>
      </div>
    )
  }

  // Show training in progress overlay
  if (trainingInProgress) {
    return (
      <div className="training-view">
        <div className="container">
          <div className="training-progress">
            <div className="spinner"></div>
            <h2>Training Model...</h2>
            <p>This may take several minutes. Please wait...</p>
            <p className="training-info">
              Training with {actualNumImages} images for {numIterations} iterations
            </p>
            <p className="training-info" style={{fontSize: '0.9rem', marginTop: '10px'}}>
              Learning rate: {learningRate} | Test images: {numTest}
            </p>
            <p className="training-info" style={{fontSize: '0.9rem', marginTop: '10px'}}>
              The page will update automatically when training completes.
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (error && images.length === 0 && !loading) {
    return (
      <div className="training-view">
        <div className="container">
          <div className="error-message">
            <p><strong>Error loading images:</strong></p>
            <p>{error}</p>
            <p style={{marginTop: '15px'}}>Make sure the backend server is running:</p>
            <code style={{display: 'block', marginTop: '10px', padding: '10px', background: '#f5f5f5', borderRadius: '4px'}}>
              cd backend && source venv/bin/activate && uvicorn main:app --reload
            </code>
          </div>
          <button onClick={onBack} className="back-btn" style={{marginTop: '20px'}}>Go Back</button>
        </div>
      </div>
    )
  }

  return (
    <div className="training-view">
      <div className="container">
        <div className="training-header">
          <button onClick={onBack} className="back-btn">← Back</button>
          <h1>Training Configuration</h1>
        </div>

        {error && !trainingStarted && (
          <div className="error-message">{error}</div>
        )}

        {trainingStarted && (
          <div className="success-message">
            Training started with {actualNumImages} images!
          </div>
        )}

        <div className="config-section">
          <div className="input-group">
            <label htmlFor="num-images">
              Number of Images to Use for Training:
            </label>
            <div className="input-with-info">
              <input
                id="num-images"
                type="number"
                min="1"
                max={maxImages}
                value={numImagesInput}
                onChange={handleNumImagesChange}
                onBlur={handleNumImagesBlur}
                disabled={trainingStarted}
                className="num-input"
                placeholder={`Enter 1-${maxImages}`}
              />
              <span className="input-info">
                (Available: {images.length} images)
              </span>
            </div>
            <div className="range-info">
              <span>Min: 1</span>
              <span>Max: {maxImages.toLocaleString()}</span>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="learning-rate">
              Learning Rate:
            </label>
            <div className="input-with-info">
              <input
                id="learning-rate"
                type="number"
                min="0.0001"
                max="1"
                step="0.0001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.005)}
                disabled={trainingStarted || trainingInProgress}
                className="num-input"
                placeholder="0.005"
              />
              <span className="input-info">
                (Recommended: 0.001 - 0.01)
              </span>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="num-iterations">
              Number of Iterations:
            </label>
            <div className="input-with-info">
              <input
                id="num-iterations"
                type="number"
                min="100"
                max="10000"
                step="100"
                value={numIterations}
                onChange={(e) => setNumIterations(parseInt(e.target.value) || 2000)}
                disabled={trainingStarted || trainingInProgress}
                className="num-input"
                placeholder="2000"
              />
              <span className="input-info">
                (More iterations = longer training, better results)
              </span>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="num-test">
              Number of Test Images:
            </label>
            <div className="input-with-info">
              <input
                id="num-test"
                type="number"
                min="100"
                max="5000"
                step="100"
                value={numTest}
                onChange={(e) => setNumTest(parseInt(e.target.value) || 1000)}
                disabled={trainingStarted || trainingInProgress}
                className="num-input"
                placeholder="1000"
              />
              <span className="input-info">
                (Available: {images.length} total)
              </span>
            </div>
            <div className="range-info">
              <span>Min: 100</span>
              <span>Max: 5,000</span>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="num-training-cats">
              Number of Training {datasetType === "brain_tumor" ? "No Tumor" : "Cats"} (Optional):
            </label>
            <div className="input-with-info">
              <input
                id="num-training-cats"
                type="number"
                min="0"
                max={maxImages}
                value={numTrainingCats === null ? '' : numTrainingCats}
                onChange={(e) => setNumTrainingCats(e.target.value === '' ? null : parseInt(e.target.value) || 0)}
                disabled={trainingStarted || trainingInProgress}
                className="num-input"
                placeholder="Auto (balanced)"
              />
              <span className="input-info">
                (Leave empty for balanced split)
              </span>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="num-training-dogs">
              Number of Training {datasetType === "brain_tumor" ? "Tumor" : "Dogs"} (Optional):
            </label>
            <div className="input-with-info">
              <input
                id="num-training-dogs"
                type="number"
                min="0"
                max={maxImages}
                value={numTrainingDogs === null ? '' : numTrainingDogs}
                onChange={(e) => setNumTrainingDogs(e.target.value === '' ? null : parseInt(e.target.value) || 0)}
                disabled={trainingStarted || trainingInProgress}
                className="num-input"
                placeholder="Auto (balanced)"
              />
              <span className="input-info">
                (Leave empty for balanced split)
              </span>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="num-test-cats">
              Number of Test {datasetType === "brain_tumor" ? "No Tumor" : "Cats"} (Optional):
            </label>
            <div className="input-with-info">
              <input
                id="num-test-cats"
                type="number"
                min="0"
                max="5000"
                value={numTestCats === null ? '' : numTestCats}
                onChange={(e) => setNumTestCats(e.target.value === '' ? null : parseInt(e.target.value) || 0)}
                disabled={trainingStarted || trainingInProgress}
                className="num-input"
                placeholder="Auto (balanced)"
              />
              <span className="input-info">
                (Leave empty for balanced split)
              </span>
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="num-test-dogs">
              Number of Test {datasetType === "brain_tumor" ? "Tumor" : "Dogs"} (Optional):
            </label>
            <div className="input-with-info">
              <input
                id="num-test-dogs"
                type="number"
                min="0"
                max="5000"
                value={numTestDogs === null ? '' : numTestDogs}
                onChange={(e) => setNumTestDogs(e.target.value === '' ? null : parseInt(e.target.value) || 0)}
                disabled={trainingStarted || trainingInProgress}
                className="num-input"
                placeholder="Auto (balanced)"
              />
              <span className="input-info">
                (Leave empty for balanced split)
              </span>
            </div>
          </div>

          {!trainingStarted && (
            <>
              {images.length === 0 && (
                <div className="no-images-message">
                  <p><strong>No dataset found!</strong></p>
                  <p>To get started:</p>
                  <ol>
                    <li>Add cat images to <code>data/raw/</code> directory</li>
                    <li>Process them: <code>cd backend && python utils/dataset_utils.py ../data/raw ../data/processed/cats.h5 hdf5</code></li>
                    <li>Refresh this page</li>
                  </ol>
                  <p>Or download from Kaggle: <a href="https://www.kaggle.com/datasets/salader/dogs-vs-cats" target="_blank" rel="noopener noreferrer">Cats vs Dogs Dataset</a></p>
                </div>
              )}
              <button 
                onClick={handleStartTraining}
                className="start-training-btn"
                disabled={
                  images.length === 0 || 
                  !actualNumImages || 
                  actualNumImages < 1 || 
                  actualNumImages > maxImages || 
                  actualNumImages + numTest > images.length ||
                  learningRate <= 0 ||
                  learningRate > 1 ||
                  numIterations < 100 ||
                  numIterations > 10000 ||
                  numTest < 100 ||
                  numTest > 5000
                }
              >
                Begin Training!
              </button>
              {images.length === 0 && (
                <p className="help-text">The button will be enabled once you have images in your dataset.</p>
              )}
            </>
          )}
        </div>

        <div className="images-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2 style={{ margin: 0 }}>
              {validNumImages > 0 ? (
                `Preview: ${startIndex + 1}-${endIndex} of ${Math.min(validNumImages, images.length)} Images`
              ) : (
                'Enter a number between 1 and 11,000 to preview images'
              )}
            </h2>
            {validNumImages > 0 && (
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <button
                onClick={handlePreviousPage}
                disabled={currentPage === 0}
                style={{
                  padding: '8px 16px',
                  fontSize: '16px',
                  cursor: currentPage === 0 ? 'not-allowed' : 'pointer',
                  opacity: currentPage === 0 ? 0.5 : 1,
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  background: 'white'
                }}
              >
                ← Previous
              </button>
              <span style={{ fontSize: '14px', color: '#666' }}>
                Page {currentPage + 1} of {totalPages}
              </span>
              <button
                onClick={handleNextPage}
                disabled={currentPage >= totalPages - 1}
                style={{
                  padding: '8px 16px',
                  fontSize: '16px',
                  cursor: currentPage >= totalPages - 1 ? 'not-allowed' : 'pointer',
                  opacity: currentPage >= totalPages - 1 ? 0.5 : 1,
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  background: 'white'
                }}
              >
                Next →
              </button>
            </div>
            )}
          </div>
          
          {validNumImages > 0 && (
          <div className="image-grid">
            {displayedImages.map((imageId, index) => {
              // Ensure imageId is used consistently (as both number and string key)
              const imageKey = imageId
              const imageKeyStr = String(imageId)
              const preview = imagePreviews[imageKey] || imagePreviews[imageKeyStr]
              
              return (
                <div key={imageId} className="image-item">
                  {preview ? (
                    <img 
                      src={preview} 
                      alt={`Cat ${startIndex + index + 1}`}
                      className="preview-image"
                      onError={(e) => {
                        console.error(`Failed to load image ${imageId}`)
                        e.target.style.display = 'none'
                        e.target.nextSibling.textContent = 'Error'
                      }}
                    />
                  ) : (
                    <div className="preview-placeholder">Loading...</div>
                  )}
                  <div className="image-label">{startIndex + index + 1}</div>
                </div>
              )
            })}
          </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TrainingView

