// src/runner.js
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

function runPythonScript(baseDir, scriptName, onLog) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(baseDir, scriptName);
    onLog(`実行開始: ${scriptName}...`, 'info');

    const py = spawn('python', ['-u', scriptPath], {
      cwd: baseDir,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1'
      }
    });

    py.stdout.on('data', (data) => onLog(data.toString('utf-8').trim()));
    py.stderr.on('data', (data) => {
      const str = data.toString('utf-8').trim();
      const isProgress = str.includes('%') || str.includes('Completed') || str.includes('Fetching') || str.includes('it/s');
      onLog(str, isProgress ? 'info' : 'error');
    });

    py.on('close', (code) => {
      if (code === 0) {
        onLog(`${scriptName} 完了 (Code: 0)`, 'success');
        resolve();
      } else {
        onLog(`${scriptName} 異常終了 (Code: ${code})`, 'error');
        reject(new Error(`Exit code ${code}`));
      }
    });
  });
}

async function runFullPipeline(baseDir, onLog) {
  await runPythonScript(baseDir, 'update_daily_features_v6.py', onLog);

  if (fs.existsSync(path.join(baseDir, 'parse_flow_signal.py'))) {
    await runPythonScript(baseDir, 'parse_flow_signal.py', onLog);
  }
  if (fs.existsSync(path.join(baseDir, 'check_sector_sentiment.py'))) {
    await runPythonScript(baseDir, 'check_sector_sentiment.py', onLog);
  }
  if (fs.existsSync(path.join(baseDir, 'fetch_margin_kabutan.py'))) {
    await runPythonScript(baseDir, 'fetch_margin_kabutan.py', onLog);
  }

  await runPythonScript(baseDir, 'run_dynamic_regime_screening_v8.py', onLog);

  if (fs.existsSync(path.join(baseDir, 'fetch_margin_kabutan.py'))) {
    await runPythonScript(baseDir, 'fetch_margin_kabutan.py', onLog);
  }
}

module.exports = { runFullPipeline };