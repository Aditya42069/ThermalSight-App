// src/main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, 'dist/index.html'));
  } else {
    mainWindow.loadURL('http://localhost:5173');
  }
}

app.whenReady().then(createWindow);

// ── helper: resolve the Python executable / script path ──────────────────────
function getBackendArgs(command, extraArgs) {
  if (app.isPackaged) {
    // PyInstaller onedir bundle: resources/backend/analyzer(.exe)
    const ext      = process.platform === 'win32' ? '.exe' : '';
    const exePath  = path.join(process.resourcesPath, 'backend', `analyzer${ext}`);
    return { executable: exePath, args: [command, ...extraArgs] };
  } else {
    // Development: run the Python script directly
    return {
      executable: 'python',
      args: [path.join(__dirname, '..', 'backend', 'analyzer.py'), command, ...extraArgs],
    };
  }
}

// ── helper: spawn + collect stdout → parse JSON ───────────────────────────────
function runPython(command, extraArgs) {
  return new Promise((resolve, reject) => {
    const { executable, args } = getBackendArgs(command, extraArgs);

    console.log(`[backend] ${executable} ${args.join(' ')}`);
    const proc = spawn(executable, args);

    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => { stdout += d.toString(); });
    proc.stderr.on('data', (d) => {
      stderr += d.toString();
      console.error(`[python] ${d.toString().trim()}`);
    });

    proc.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(stdout.trim()));
        } catch (e) {
          reject(`JSON parse error: ${e.message}\nRaw output: ${stdout}`);
        }
      } else {
        // Try to parse the error JSON the script emits on failure
        try {
          const errObj = JSON.parse(stdout.trim());
          reject(errObj.error || `Process exited with code ${code}`);
        } catch {
          reject(`Process exited with code ${code}.\nStderr: ${stderr}`);
        }
      }
    });

    proc.on('error', (err) => {
      reject(`Failed to start backend: ${err.message}`);
    });
  });
}

// ── IPC: run-analysis ─────────────────────────────────────────────────────────
// Called by React with: ipcRenderer.invoke('run-analysis', imagePath, outputDir)
// Returns JSON: { status, stem, out_dir, shape, temp_min, temp_max, images, csvs }
ipcMain.handle('run-analysis', async (_event, imagePath, outputDir) => {
  return runPython('analyze', [imagePath, outputDir]);
});

// ── IPC: measure-roi ──────────────────────────────────────────────────────────
// Called by React with:
//   ipcRenderer.invoke('measure-roi', imagePath, cx, cy, r_cm, pxPerCm, outputDir)
// Returns JSON: { status, csv_path, directional, stats, dominant, grad_per_cm, … }
ipcMain.handle('measure-roi', async (_event, imagePath, cx, cy, r_cm, pxPerCm, outputDir) => {
  return runPython('roi', [
    imagePath,
    String(Math.round(cx)),
    String(Math.round(cy)),
    String(r_cm),
    String(pxPerCm),
    outputDir,
  ]);
});

// ── IPC: open-file-dialog ─────────────────────────────────────────────────────
// Convenience so React can open a native file picker without nodeIntegration hacks
ipcMain.handle('open-file-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select FLIR / thermal image',
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'tiff', 'tif'] },
      { name: 'All files', extensions: ['*'] },
    ],
    properties: ['openFile'],
  });
  return result.canceled ? null : result.filePaths[0];
});

// ── IPC: open-folder-dialog ───────────────────────────────────────────────────
ipcMain.handle('open-folder-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select output folder',
    properties: ['openDirectory', 'createDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});
