import { useState, useEffect } from 'react'
import axios from 'axios'
import '../../styles/TrainingView.css'

// Configure axios to use backend URL
const API_BASE_URL = 'http://localhost:8000'
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000
})

function TrainingView({ onBack }) {
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [numImages, setNumImages] = useState(null)
  const [numImagesInput, setNumImagesInput] = useState('')
  const [trainingStarted, setTrainingStarted] = useState(false)
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
        
        // If we have total_count but no images array, generate indices up to 11,000
        if (totalCount > 0 && imageList.length === 0) {
          const maxToGenerate = Math.min(11000, totalCount)
          imageList = Array.from({ length: maxToGenerate }, (_, i) => i)
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
  }, [])

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
        const response = await api.get(`/api/images/preview/${imageIdStr}?size=64`, {
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
    // Clamp to valid range (2000-11000) only on blur
    if (value < 2000) {
      setNumImages(2000)
      setNumImagesInput('2000')
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
    
    if (!actualNumImages || actualNumImages < 2000 || actualNumImages > 11000) {
      setError('Number of images must be between 2,000 and 11,000')
      return
    }

    if (actualNumImages > images.length) {
      setError(`Not enough images. Available: ${images.length}, Requested: ${actualNumImages}`)
      return
    }

    try {
      setError(null)
      const response = await api.post('/api/training/start', {
        num_images: actualNumImages
      })
      setTrainingStarted(true)
      console.log('Training started:', response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start training')
      console.error('Error starting training:', err)
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
                min="2000"
                max="11000"
                value={numImagesInput}
                onChange={handleNumImagesChange}
                onBlur={handleNumImagesBlur}
                disabled={trainingStarted}
                className="num-input"
                placeholder="Enter 2000-11000"
              />
              <span className="input-info">
                (Available: {images.length} images)
              </span>
            </div>
            <div className="range-info">
              <span>Min: 2,000</span>
              <span>Max: 11,000 (1,000 reserved for testing)</span>
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
                disabled={images.length === 0 || !actualNumImages || actualNumImages < 2000 || actualNumImages > 11000 || actualNumImages > images.length}
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
                'Enter a number between 2,000 and 11,000 to preview images'
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

