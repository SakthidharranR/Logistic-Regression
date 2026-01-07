import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { useState } from 'react'
import PredictionVisualization from './PredictionVisualization'
import '../../styles/ResultsView.css'

function ResultsView({ results, onRetrain, onBack, onClearResults }) {
  const [expandedDescriptions, setExpandedDescriptions] = useState({})
  const [showVisualization, setShowVisualization] = useState(false)
  
  // Handle both single result (backward compatibility) and array of results
  const resultsArray = Array.isArray(results) ? results : [results]
  const latestResult = resultsArray[resultsArray.length - 1]
  
  const toggleDescription = (chartId) => {
    setExpandedDescriptions(prev => ({
      ...prev,
      [chartId]: !prev[chartId]
    }))
  }

  // Prepare data for all results
  const allResultsData = resultsArray.map((result, index) => {
    // Costs are recorded every 100 iterations (at iterations 0, 100, 200, 300, etc.)
    // So cost at index 0 is at iteration 0, index 1 at 100, index 2 at 200, etc.
    const costData = result.costs.map((cost, idx) => {
      // Each cost point represents 100 iterations
      const iteration = idx * 100
      return {
        iteration: iteration,
        cost: cost
      }
    })

    // Prepare accuracy data with run-specific keys for grouped bars
    const accuracyData = [
      { 
        name: 'Training', 
        accuracy: result.train_accuracy,
        [`run${index + 1}Accuracy`]: result.train_accuracy
      },
      { 
        name: 'Testing', 
        accuracy: result.test_accuracy,
        [`run${index + 1}Accuracy`]: result.test_accuracy
      }
    ]

    return {
      ...result,
      costData,
      accuracyData,
      trainingDistribution: [
        { name: result.dataset_type === "brain_tumor" ? "No Tumor" : "Cats", count: result.training_cats || 0 },
        { name: result.dataset_type === "brain_tumor" ? "Tumor" : "Dogs", count: result.training_dogs || 0 }
      ],
      testDistribution: [
        { name: result.dataset_type === "brain_tumor" ? "No Tumor" : "Cats", count: result.test_cats || 0 },
        { name: result.dataset_type === "brain_tumor" ? "Tumor" : "Dogs", count: result.test_dogs || 0 }
      ]
    }
  })

  // Merge all accuracy data for grouped bar chart
  const mergedAccuracyData = allResultsData.length > 0 ? [
    {
      name: 'Training',
      ...allResultsData.reduce((acc, resultData, idx) => {
        acc[`run${idx + 1}Accuracy`] = resultData.train_accuracy
        return acc
      }, {})
    },
    {
      name: 'Testing',
      ...allResultsData.reduce((acc, resultData, idx) => {
        acc[`run${idx + 1}Accuracy`] = resultData.test_accuracy
        return acc
      }, {})
    }
  ] : []

  // Use latest result for single-result displays
  const latestData = allResultsData[allResultsData.length - 1]
  const costData = latestData.costData
  const accuracyData = latestData.accuracyData
  const trainingDistribution = latestData.trainingDistribution
  const testDistribution = latestData.testDistribution
  
  // Determine dataset type and labels
  const datasetType = latestResult.dataset_type || "cats_dogs"
  const class0Label = datasetType === "brain_tumor" ? "No Tumor" : "Cats"
  const class1Label = datasetType === "brain_tumor" ? "Tumor" : "Dogs"

  if (showVisualization) {
    return (
      <PredictionVisualization
        result={latestResult}
        datasetType={datasetType}
        onBack={() => setShowVisualization(false)}
      />
    )
  }

  return (
    <div className="results-view">
      <div className="container">
        <div className="results-header">
          <button onClick={onBack} className="back-btn">← Back</button>
          <h1>Training Results {resultsArray.length > 1 && `(${resultsArray.length} runs)`}</h1>
          {resultsArray.length > 1 && onClearResults && (
            <button onClick={onClearResults} className="clear-results-btn">Clear All Results</button>
          )}
        </div>

        {/* Comparison Summary - Show all runs side by side */}
        {resultsArray.length > 1 && (
          <div className="comparison-section">
            <h2>Comparison of All Training Runs</h2>
            <div className="comparison-grid">
              {allResultsData.map((resultData, index) => (
                <div key={index} className="comparison-card">
                  <div className="comparison-header">
                    <h3>Run #{resultData.runNumber || index + 1}</h3>
                    {resultData.timestamp && (
                      <span className="run-time">
                        {new Date(resultData.timestamp).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                  <div className="comparison-metrics">
                    <div className="comparison-metric">
                      <span className="metric-label">Train Acc:</span>
                      <span className="metric-value">{resultData.train_accuracy.toFixed(2)}%</span>
                    </div>
                    <div className="comparison-metric">
                      <span className="metric-label">Test Acc:</span>
                      <span className="metric-value">{resultData.test_accuracy.toFixed(2)}%</span>
                    </div>
                    <div className="comparison-metric">
                      <span className="metric-label">LR:</span>
                      <span className="metric-value">{resultData.learning_rate}</span>
                    </div>
                    <div className="comparison-metric">
                      <span className="metric-label">Iterations:</span>
                      <span className="metric-value">{resultData.num_iterations}</span>
                    </div>
                    <div className="comparison-metric">
                      <span className="metric-label">Train:</span>
                      <span className="metric-value">{resultData.training_set_size}</span>
                    </div>
                    <div className="comparison-metric">
                      <span className="metric-label">Test:</span>
                      <span className="metric-value">{resultData.test_set_size}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Summary Cards - Latest Run */}
        <div className="summary-cards">
          <div className="summary-card">
            <h3>Training Accuracy {resultsArray.length > 1 && '(Latest)'}</h3>
            <div className="metric-value">{latestResult.train_accuracy.toFixed(2)}%</div>
          </div>
          <div className="summary-card">
            <h3>Test Accuracy {resultsArray.length > 1 && '(Latest)'}</h3>
            <div className="metric-value">{latestResult.test_accuracy.toFixed(2)}%</div>
          </div>
          <div className="summary-card">
            <h3>Training Set</h3>
            <div className="metric-value">{latestResult.training_set_size} images</div>
            <div className="metric-detail">
              {latestResult.training_cats || 0} {class0Label}, {latestResult.training_dogs || 0} {class1Label}
            </div>
          </div>
          <div className="summary-card">
            <h3>Test Set</h3>
            <div className="metric-value">{latestResult.test_set_size} images</div>
            <div className="metric-detail">
              {latestResult.test_cats || 0} {class0Label}, {latestResult.test_dogs || 0} {class1Label}
            </div>
          </div>
        </div>

        {/* Hyperparameters */}
        <div className="hyperparams-section">
          <h2>Hyperparameters & Dataset Info</h2>
          <div className="hyperparams-grid">
            <div className="hyperparam-item">
              <span className="label">Learning Rate:</span>
              <span className="value">{latestResult.learning_rate}</span>
            </div>
            <div className="hyperparam-item">
              <span className="label">Iterations:</span>
              <span className="value">{latestResult.num_iterations}</span>
            </div>
            <div className="hyperparam-item">
              <span className="label">Training Images:</span>
              <span className="value">{latestResult.training_set_size}</span>
            </div>
            <div className="hyperparam-item">
              <span className="label">Test Images:</span>
              <span className="value">{latestResult.test_set_size}</span>
            </div>
            <div className="hyperparam-item">
              <span className="label">Training {class0Label}:</span>
              <span className="value">{latestResult.training_cats || 0}</span>
            </div>
            <div className="hyperparam-item">
              <span className="label">Training {class1Label}:</span>
              <span className="value">{latestResult.training_dogs || 0}</span>
            </div>
            <div className="hyperparam-item">
              <span className="label">Test {class0Label}:</span>
              <span className="value">{latestResult.test_cats || 0}</span>
            </div>
            <div className="hyperparam-item">
              <span className="label">Test {class1Label}:</span>
              <span className="value">{latestResult.test_dogs || 0}</span>
            </div>
          </div>
        </div>

        {/* Cost Over Time Chart - Comparison */}
        <div className="chart-section">
          <div className="chart-header">
            <h2>Training Cost Over Time {resultsArray.length > 1 && '(All Runs)'}</h2>
            <button 
              className="info-toggle"
              onClick={() => toggleDescription('cost')}
              title="Show/hide description"
            >
              ℹ️
            </button>
          </div>
          {expandedDescriptions.cost && (
            <p className="chart-description">
              This graph shows how the model's error (cost) decreases during training. A decreasing line means the model is learning. 
              <strong> Lower learning rates</strong> create smoother, gradual decreases. <strong>Higher learning rates</strong> may cause faster drops but can be unstable. 
              <strong>More iterations</strong> allow the cost to decrease further, potentially improving accuracy.
            </p>
          )}
          <ResponsiveContainer width="100%" height={300}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="iteration" 
                label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                label={{ value: 'Cost', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Legend />
              {allResultsData.map((resultData, index) => {
                const colors = ['#667eea', '#48bb78', '#ed8936', '#9f7aea', '#f56565']
                const color = colors[index % colors.length]
                return (
                  <Line 
                    key={index}
                    type="monotone" 
                    dataKey="cost" 
                    data={resultData.costData}
                    stroke={color} 
                    strokeWidth={2} 
                    name={`Run ${resultData.runNumber || index + 1}`}
                    dot={false}
                  />
                )
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Accuracy Comparison Chart - All Runs */}
        <div className="chart-section">
          <div className="chart-header">
            <h2>Accuracy Comparison {resultsArray.length > 1 && '(All Runs)'}</h2>
            <button 
              className="info-toggle"
              onClick={() => toggleDescription('accuracy')}
              title="Show/hide description"
            >
              ℹ️
            </button>
          </div>
          {expandedDescriptions.accuracy && (
            <p className="chart-description">
              This compares how well the model performs on training data vs. test data. Training accuracy shows how well the model learned the training examples. 
              Test accuracy shows how well it generalizes to new data. A <strong>large gap</strong> between them suggests overfitting. 
              <strong>More training images</strong> typically improve both accuracies. <strong>More iterations</strong> can increase training accuracy but may overfit if test accuracy doesn't improve.
            </p>
          )}
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mergedAccuracyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              {allResultsData.map((resultData, index) => {
                const colors = ['#667eea', '#48bb78', '#ed8936', '#9f7aea', '#f56565']
                const color = colors[index % colors.length]
                return (
                  <Bar 
                    key={index}
                    dataKey={`run${index + 1}Accuracy`}
                    fill={color} 
                    name={`Run ${resultData.runNumber || index + 1}`}
                  />
                )
              })}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Dataset Distribution Charts */}
        <div className="charts-row">
          <div className="chart-section">
            <div className="chart-header">
              <h2>Training Set Distribution</h2>
              <button 
                className="info-toggle"
                onClick={() => toggleDescription('trainingDist')}
                title="Show/hide description"
              >
                ℹ️
              </button>
            </div>
            {expandedDescriptions.trainingDist && (
              <p className="chart-description">
                Shows the balance of {class0Label} vs. {class1Label} in your training data. A <strong>balanced dataset</strong> (similar counts) helps the model learn both classes equally. 
                An <strong>imbalanced dataset</strong> may cause the model to favor the more common class.
              </p>
            )}
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={trainingDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#48bb78" name="Images" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-section">
            <div className="chart-header">
              <h2>Test Set Distribution</h2>
              <button 
                className="info-toggle"
                onClick={() => toggleDescription('testDist')}
                title="Show/hide description"
              >
                ℹ️
              </button>
            </div>
            {expandedDescriptions.testDist && (
              <p className="chart-description">
                Shows the balance of {class0Label} vs. {class1Label} in your test data. This should be similar to the training distribution for fair evaluation. 
                The test set size is controlled by the <strong>"Number of Test Images"</strong> parameter.
              </p>
            )}
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={testDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#ed8936" name="Images" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="results-actions">
          <button 
            onClick={() => setShowVisualization(true)} 
            className="visualize-btn"
          >
            View Predictions (Correct/Incorrect Images)
          </button>
          <button onClick={onRetrain} className="retrain-btn">
            Train Again with Different Parameters
          </button>
        </div>
      </div>
    </div>
  )
}

export default ResultsView

