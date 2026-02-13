// Preload script — runs in renderer context with context isolation
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  selectModelFolder: () => ipcRenderer.invoke('select-model-folder'),
});
