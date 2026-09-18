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

// 日次自動実行対象のスクリプトは pipeline/ 配下に置いている(役割別フォルダ整理)
const PIPELINE_DIR = 'pipeline';

async function runFullPipeline(baseDir, onLog) {
  // 株価・信用残・日経225等のJ-Quantsキャッシュを最新化(run_dynamic_regime_screening_v8.py
  // が必須で読みに行くため、必ず最初に完了させておく必要がある)
  await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'build_jquants_cache.py'), onLog);

  await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'update_daily_features_v6.py'), onLog);

  // VIX/USD-JPY(地合い危険度モデルの特徴量)を更新。更新を怠るとffill()で古い値を
  // 使い続けてしまいエラーが出ないため、必ず日次で実行する。
  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'update_fx_vix_cache.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'update_fx_vix_cache.py'), onLog);
  }

  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'parse_flow_signal.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'parse_flow_signal.py'), onLog);
  }
  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'check_sector_sentiment.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'check_sector_sentiment.py'), onLog);
  }

  await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'run_dynamic_regime_screening_v8.py'), onLog);

  // 本日のポートフォリオ採用銘柄をトラッキングログに記録し、保留中の過去の推奨を
  // 本日の値動きで判定する(利確/損切/タイムアウト)。実運用でしか得られない
  // 本当の意味でのアウトオブサンプル検証を蓄積するための機能。
  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'update_tracking_log.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'update_tracking_log.py'), onLog);
  }

  // 各種日次キャッシュの抜け漏れを埋める安全網。既存日はスキップする作りなので
  // 通常はほぼ何もせず一瞬で終わる(実行を忘れた日があった場合だけ効く)。
  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'backfill_sector_sentiment.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'backfill_sector_sentiment.py'), onLog);
  }
  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'backfill_regime_risk_cache.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'backfill_regime_risk_cache.py'), onLog);
  }
  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'backfill_flow_signal_cache.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'backfill_flow_signal_cache.py'), onLog);
  }
  // 空売り機会モデル(short_edge)の学習特徴量用。まだライブスクリーニングの判定には
  // 使っていないが、日々キャッシュを伸ばしておくことで再学習時の対象期間が広がる。
  if (fs.existsSync(path.join(baseDir, PIPELINE_DIR, 'backfill_short_ratio_cache.py'))) {
    await runPythonScript(baseDir, path.join(PIPELINE_DIR, 'backfill_short_ratio_cache.py'), onLog);
  }
}

module.exports = { runFullPipeline };