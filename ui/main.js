const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { getTickerBars } = require('./src/daily_bars_main');

// daily_screening_bars_raw.parquet(縦持ちOHLCV、build_jquants_cache.py生成)の絶対パス。
// hyparquetでのパースはメインプロセス側(src/daily_bars_main.js)で行う(下記コメント参照)。
const DAILY_BARS_PARQUET_PATH = path.join(__dirname, '..', 'data', 'cache', 'daily_screening_bars_raw.parquet');

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    title: "JPX Regime Swing Screener",
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  win.loadFile('index.html');

  // エラー原因を即座に確認できるようにConsoleを開く
  win.webContents.openDevTools();
}

// レンダラー(nodeIntegration:trueのclassicスクリプト)側で直接 import('hyparquet') すると、
// Node.jsの動的importではなくChromiumページ側のESMローダーにルーティングされてしまい、
// 二度と解決しないPromiseのままフリーズする不具合があった(2026-09-24、「UIがまるっきり
// 表示されない」で発覚。レンダラーは company_master 等の同期requireまでは正常に動き、
// import('hyparquet') の直後で完全に固まっていた)。
// メインプロセスは純粋なNode.js環境で動的importが確実に動くため、パース処理をこちらに
// 移し、IPC経由でレンダラーに銘柄別の足データだけを返す構成にした。
ipcMain.handle('get-ticker-bars', async (event, dotCode, rawCode) => {
  try {
    return await getTickerBars(DAILY_BARS_PARQUET_PATH, dotCode, rawCode);
  } catch (e) {
    console.error('[main] get-ticker-bars failed:', e);
    return null;
  }
});

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
