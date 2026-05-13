// src/main.js
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1024,
    height: 768,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false // For simplicity in this tutorial
    }
  });

  // In development, load the React local server. In production, load the built HTML.
  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, 'dist/index.html'));
  } else {
    mainWindow.loadURL('http://localhost:5173');
  }
}

app.whenReady().then(createWindow);

// This listens for React asking to run the Python analysis
ipcMain.handle('run-analysis', async (event, imagePath, outputDir) => {
  return new Promise((resolve, reject) => {
    // Determine where the Python executable is located
    let executablePath;
    if (app.isPackaged) {
        // In the final app, it will be bundled here
        executablePath = path.join(process.resourcesPath, 'backend', 'analyzer.exe'); 
        // Note: on Mac/Linux, omit the .exe extension
    } else {
        // In development, we just run the python script directly
        executablePath = 'python'; 
    }

    const args = app.isPackaged 
        ? [imagePath, outputDir] 
        : ['../backend/analyzer.py', imagePath, outputDir];

    const pythonProcess = spawn(executablePath, args);

    let outputData = '';
    pythonProcess.stdout.on('data', (data) => { outputData += data.toString(); });
    pythonProcess.stderr.on('data', (data) => { console.error(`Error: ${data}`); });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(JSON.parse(outputData)); // Send the JSON response back to React
      } else {
        reject('Analysis failed');
      }
    });
  });
});