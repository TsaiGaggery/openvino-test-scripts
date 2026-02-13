const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');
const fs = require('fs');

const isDev = !app.isPackaged;
const isWin = process.platform === 'win32';

// Project root is one level up from benchmark-studio/
const projectDir = isDev
  ? path.resolve(__dirname, '..')
  : path.join(process.resourcesPath, 'backend');

const studioDir = isDev
  ? __dirname
  : path.join(process.resourcesPath, 'backend', 'benchmark-studio');

const PORT = 8085;
const SERVER_URL = `http://localhost:${PORT}`;

let flaskProcess = null;
let mainWindow = null;

function startFlaskServer() {
  const command = isWin ? 'python' : 'python3';
  const args = [path.join(studioDir, 'server.py')];

  const env = Object.assign({}, process.env, {
    BENCHMARK_PROJECT_DIR: projectDir,
    BENCHMARK_STUDIO_PORT: String(PORT),
  });

  flaskProcess = spawn(command, args, {
    cwd: projectDir,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: env,
  });

  flaskProcess.stdout.on('data', (data) => {
    process.stdout.write(`[server] ${data}`);
  });

  flaskProcess.stderr.on('data', (data) => {
    process.stderr.write(`[server] ${data}`);
  });

  flaskProcess.on('close', (code) => {
    console.log(`[server] process exited with code ${code}`);
    flaskProcess = null;
  });

  flaskProcess.on('error', (err) => {
    console.error(`[server] failed to start: ${err.message}`);
    flaskProcess = null;
  });
}

function waitForServer(maxRetries = 60, interval = 500) {
  return new Promise((resolve, reject) => {
    let retries = 0;

    const check = () => {
      const req = http.get(`${SERVER_URL}/api/health`, (res) => {
        if (res.statusCode === 200) {
          resolve();
        } else {
          retry();
        }
      });

      req.on('error', () => retry());
      req.setTimeout(1000, () => {
        req.destroy();
        retry();
      });
    };

    const retry = () => {
      retries++;
      if (retries >= maxRetries) {
        reject(new Error('Flask server did not start in time'));
      } else {
        setTimeout(check, interval);
      }
    };

    check();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: 'OpenVINO Benchmark Studio',
    backgroundColor: '#0B0F19',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(SERVER_URL);
  mainWindow.setMenuBarVisibility(false);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function stopFlaskServer() {
  if (!flaskProcess) return Promise.resolve();

  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      if (flaskProcess) {
        console.log('[server] force-killing after timeout');
        if (isWin) {
          spawn('taskkill', ['/pid', flaskProcess.pid.toString(), '/f', '/t']);
        } else {
          flaskProcess.kill('SIGKILL');
        }
      }
      resolve();
    }, 5000);

    flaskProcess.on('close', () => {
      clearTimeout(timeout);
      resolve();
    });

    console.log('[server] stopping server');
    if (isWin) {
      spawn('taskkill', ['/pid', flaskProcess.pid.toString(), '/t']);
    } else {
      flaskProcess.kill('SIGTERM');
    }
  });
}

ipcMain.handle('select-model-folder', async () => {
  if (!mainWindow) return { canceled: true };

  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select OpenVINO Model Folder',
    properties: ['openDirectory'],
    message: 'Choose a folder containing OpenVINO model files (.xml and .bin)',
  });

  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true };
  }

  return { canceled: false, folderPath: result.filePaths[0] };
});

app.whenReady().then(async () => {
  console.log(`Mode: ${isDev ? 'development' : 'packaged'}`);
  console.log(`Project dir: ${projectDir}`);
  console.log(`Studio dir: ${studioDir}`);

  console.log('Starting Flask server...');
  startFlaskServer();

  try {
    await waitForServer();
    console.log('Flask server is ready');
    createWindow();
  } catch (err) {
    console.error(err.message);
    app.quit();
  }
});

app.on('window-all-closed', async () => {
  await stopFlaskServer();
  app.quit();
});

app.on('before-quit', async (event) => {
  if (flaskProcess) {
    event.preventDefault();
    await stopFlaskServer();
    app.quit();
  }
});
