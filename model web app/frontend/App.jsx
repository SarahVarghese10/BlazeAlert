import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import RiskGauge from './components/RiskGauge';
import FactorsBreakdown from './components/FactorsBreakdown';
import RiskMap from './components/RiskMap';
import TestingTab from './components/TestingTab';
import './App.css';

function App() {
  const [zipcode, setZipcode] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('predict'); // 'predict' or 'testing'

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!zipcode || zipcode.length !== 5) {
      setError('Please enter a valid 5-digit zipcode');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ zipcode }),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="app-container">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="header"
        >
          <h1 className="title glow">BlazeAlert</h1>
          <p className="subtitle">A Novel Spatiotemporal Deep Convolutional Neural Network Model for Real Time Wildfire Risk Assessment Harnessing Historical Geospatial Data</p>
        </motion.header>

        {/* Tab Navigation */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="tab-navigation"
        >
          <button
            className={`tab-button ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            Live Prediction
          </button>
          <button
            className={`tab-button ${activeTab === 'testing' ? 'active' : ''}`}
            onClick={() => setActiveTab('testing')}
          >
            Historical Test Cases
          </button>
        </motion.div>

        {/* Conditional Content Based on Active Tab */}
        {activeTab === 'predict' ? (
          <>
        {/* Search Form */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="search-section"
        >
          <form onSubmit={handlePredict} className="search-form">
            <div className="input-wrapper">
              <input
                type="text"
                value={zipcode}
                onChange={(e) => setZipcode(e.target.value.replace(/\D/g, '').slice(0, 5))}
                placeholder="Enter ZIP Code"
                className="zipcode-input"
                maxLength="5"
              />
              <button 
                type="submit" 
                className="predict-button"
                disabled={loading}
              >
                {loading ? (
                  <span className="loading-spinner"></span>
                ) : (
                  'ANALYZE RISK'
                )}
              </button>
            </div>
          </form>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="error-message"
            >
              ⚠️ {error}
            </motion.div>
          )}
        </motion.div>

        {/* Results */}
        <AnimatePresence mode="wait">
          {prediction && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50 }}
              transition={{ duration: 0.6 }}
              className="results-container"
            >
              {/* Risk Gauge */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                className="gauge-section"
              >
                <RiskGauge 
                  score={prediction.risk_score} 
                  level={prediction.risk_level}
                />
              </motion.div>

              {/* Location Info */}
              <motion.div
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3, duration: 0.5 }}
                className="location-info"
              >
                <h3>Location Analysis</h3>
                <p>Coordinates: {prediction.latitude.toFixed(4)}°N, {prediction.longitude.toFixed(4)}°W</p>
                <p className="timestamp">Updated: {new Date(prediction.timestamp).toLocaleString()}</p>
              </motion.div>

              {/* Contributing Factors */}
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
                className="factors-section"
              >
                <FactorsBreakdown factors={prediction.contributing_factors} />
              </motion.div>

              {/* Map */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.5 }}
                className="map-section"
              >
                <RiskMap 
                  latitude={prediction.latitude}
                  longitude={prediction.longitude}
                  riskLevel={prediction.risk_level}
                />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty State */}
        {!prediction && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="empty-state"
          >
            <h2>Enter a ZIP code to begin analysis</h2>
          </motion.div>
        )}
      </>
        ) : (
          <TestingTab />
        )}
      </div>
    </div>
  );
}

export default App;
