import React, { useState } from 'react';
import './App.css';

// 1. We define ipcRenderer up here at the top
const { ipcRenderer } = window.require('electron');

function App() {
  const [filePath, setFilePath] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [viewMode, setViewMode] = useState('quiver'); // 'quiver' or 'gradient'

  const handleFileDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) setFilePath(file.path);
  };

  // 2. THIS IS YOUR NEW FUNCTION
  const runAnalysis = async () => {
    setIsProcessing(true);
    try {
      // I updated the output path to save the folder right next to the original image!
      const outputDir = filePath + '_analysis_results'; 
      
      // Call the Python script via Electron's backend
      const response = await ipcRenderer.invoke('run-analysis', filePath, outputDir);
      
      setResults(response);
    } catch (error) {
      console.error("Failed:", error);
      alert("Something went wrong during analysis. Check the console.");
    }
    setIsProcessing(false);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>ThermalSight Student Explorer</h1>
        <button className="help-btn">? Help</button>
      </header>

      {!results ? (
        <main className="upload-screen">
          <div 
            className="drop-zone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleFileDrop}
          >
            {filePath ? (
              <p>Selected: {filePath}</p>
            ) : (
              <p>Drag & Drop a thermal image here (.jpg, .png)</p>
            )}
          </div>
          
          <button 
            className="analyze-btn" 
            disabled={!filePath || isProcessing}
            onClick={runAnalysis}
          >
            {isProcessing ? 'Analyzing Heat Flow...' : 'Analyze Image'}
          </button>
        </main>
      ) : (
        <main className="results-screen">
          <div className="tabs">
            <button 
              className={viewMode === 'quiver' ? 'active' : ''} 
              onClick={() => setViewMode('quiver')}
            >
              Heat Direction (Arrows)
            </button>
            <button 
              className={viewMode === 'gradient' ? 'active' : ''} 
              onClick={() => setViewMode('gradient')}
            >
              Temperature Gradient
            </button>
          </div>
          
          <div className="image-viewer">
            <div className="placeholder-image">
              {/* In a fully finished app, this would display the actual image from the hard drive */}
              [Image saved to: {results.output_dir}]
              <br/><br/>
              [Showing: {viewMode === 'quiver' ? 'Quiver Plot' : 'Gradient Map'}]
            </div>
          </div>
          
          <div className="educational-context">
            <h3>What are you looking at?</h3>
            <p>
              {viewMode === 'quiver' 
                ? "These arrows show the direction heat is flowing. Larger arrows mean a faster temperature change!" 
                : "This map highlights the areas with the most extreme temperature shifts. Brighter colors equal sharper changes."}
            </p>
            <button className="export-btn">Open Exported Data (CSV)</button>
          </div>
          
          <button className="reset-btn" onClick={() => setResults(null)}>Start Over</button>
        </main>
      )}
    </div>
  );
}

export default App;