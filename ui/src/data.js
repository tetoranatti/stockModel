// src/data.js
const fs = require('fs');
const path = require('path');

function parseCsvLine(text) {
  const result = [];
  let cur = '';
  let inQuote = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (inQuote && text[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuote = !inQuote;
      }
    } else if (c === ',' && !inQuote) {
      result.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  result.push(cur);
  return result;
}

function loadScreenedCsv(baseDir) {
  const targetCsv = path.join(baseDir, 'final_regime_screened_v8.csv');
  if (!fs.existsSync(targetCsv)) return { filename: '', records: [] };

  const raw = fs.readFileSync(targetCsv, 'utf-8');
  const lines = raw.trim().split(/\r?\n/);
  if (lines.length <= 1) return { filename: path.basename(targetCsv), records: [] };

  const records = [];
  const headers = parseCsvLine(lines[0]).map(h => h.trim());
  const getIdx = (name) => headers.indexOf(name);

  const idxTicker = getIdx('ticker');
  const idxPrice = getIdx('price');
  const idxTarget = getIdx('target_price') !== -1 ? getIdx('target_price') : getIdx('target');
  const idxStop = getIdx('stop_price') !== -1 ? getIdx('stop_price') : getIdx('stop');
  const idxPWin = getIdx('prob_win') !== -1 ? getIdx('prob_win') : getIdx('p_win');
  const idxPWinRaw = getIdx('prob_win_raw');
  const idxPStop = getIdx('prob_stop') !== -1 ? getIdx('prob_stop') : getIdx('p_stop');
  const idxEv = getIdx('ev_score') !== -1 ? getIdx('ev_score') : getIdx('ev');
  const idxEvRaw = getIdx('ev_score_raw');
  const idxBeta = getIdx('beta');
  const idxVol = getIdx('vol_ratio') !== -1 ? getIdx('vol_ratio') : getIdx('vol');
  const idxTurnover = getIdx('turnover_oku') !== -1 ? getIdx('turnover_oku') : getIdx('turnover');
  const idxDaysToClear = getIdx('days_to_clear');
  const idxSizeFactor = getIdx('size_factor');
  const idxSizeReason = getIdx('size_reason');
  const idxAction = getIdx('action');
  const idxReason = getIdx('reason');
  const idxRatio = getIdx('margin_ratio');
  const idxBuyPct = getIdx('margin_buy_pct');
  const idxAcc = getIdx('is_accumulating');
  const idxSecName = getIdx('sector_name');
  const idxSecScore = getIdx('sector_score');
  const idxSecShock = getIdx('sector_shock');
  const idxSecAdvice = getIdx('sector_advice');
  const idxSecSummary = getIdx('sector_summary');

  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const row = parseCsvLine(lines[i]);
    if (row.length < 8) continue;

    const ticker = (row[idxTicker] || '').trim();
    if (!ticker) continue;

    records.push({
      ticker,
      companyName: '',
      price: (row[idxPrice] || '').trim(),
      target: (row[idxTarget] || '').trim(),
      stop: (row[idxStop] || '').trim(),
      p_win: (row[idxPWin] || '').trim(),
      p_win_raw: (idxPWinRaw !== -1 && row[idxPWinRaw]) ? row[idxPWinRaw].trim() : (row[idxPWin] || '').trim(),
      p_stop: (row[idxPStop] || '').trim(),
      ev: (row[idxEv] || '').trim(),
      ev_raw: (idxEvRaw !== -1 && row[idxEvRaw]) ? row[idxEvRaw].trim() : (row[idxEv] || '').trim(),
      beta: (row[idxBeta] || '').trim(),
      vol: (row[idxVol] || '').trim(),
      turnover: (row[idxTurnover] || '').trim(),
      days_to_clear: (idxDaysToClear !== -1 && row[idxDaysToClear]) ? row[idxDaysToClear].trim() : '',
      size_factor: (idxSizeFactor !== -1 && row[idxSizeFactor]) ? row[idxSizeFactor].trim() : '1.0',
      size_reason: (idxSizeReason !== -1 && row[idxSizeReason]) ? row[idxSizeReason].trim() : '',
      action: (row[idxAction] || '').trim(),
      reason: (row[idxReason] || '').trim(),
      margin_ratio: (idxRatio !== -1 && row[idxRatio]) ? row[idxRatio].trim() : '',
      margin_buy_pct: (idxBuyPct !== -1 && row[idxBuyPct]) ? row[idxBuyPct].trim() : '0.0',
      is_accumulating: (idxAcc !== -1 && row[idxAcc]) ? row[idxAcc].trim() : 'False',
      sector_name: (idxSecName !== -1 && row[idxSecName]) ? row[idxSecName].trim() : '',
      sector_score: (idxSecScore !== -1 && row[idxSecScore]) ? parseFloat(row[idxSecScore]) || 0.0 : 0.0,
      sector_shock: (idxSecShock !== -1 && (row[idxSecShock] === 'True' || row[idxSecShock] === 'true')),
      sector_advice: (idxSecAdvice !== -1 && row[idxSecAdvice]) ? row[idxSecAdvice].trim() : '通常',
      sector_summary: (idxSecSummary !== -1 && row[idxSecSummary]) ? row[idxSecSummary].trim() : '',
    });
  }

  return { filename: path.basename(targetCsv), records };
}

function loadMacroFlowSignal(baseDir) {
  const jsonPath = path.join(baseDir, 'macro_flow_signal.json');
  if (!fs.existsSync(jsonPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
  } catch (e) {
    return null;
  }
}

// run_dynamic_regime_screening_v8.py が出力する data/screening_results_v8.json を読む。
// 実際にK(採用銘柄数)・サイズ倍率を左右している地合い危険度モデルの出力(regime_risk)は
// このファイルにしかないため、CSVとは別に取得する。
function loadScreeningMeta(baseDir) {
  const jsonPath = path.join(baseDir, 'data', 'screening_results_v8.json');
  if (!fs.existsSync(jsonPath)) return null;
  try {
    const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    return {
      updatedAt: data.updated_at || '',
      macroRegime: data.macro_regime || null,
      regimeRisk: data.regime_risk || null,
      shortEdge: data.short_edge || null,
    };
  } catch (e) {
    return null;
  }
}

// run_dynamic_regime_screening_v8.py が出力する data/portfolio_latest.json を読む。
// 複数銘柄を同時に買う前提で資金・信用枠を共有配分したポートフォリオ(build_portfolio()の
// 結果)。個別銘柄ごとのスタンドアロン試算(final_regime_screened_v8.csvのstandalone_*列)
// とは異なり、こちらが実際に発注すべき株数の最終値。
function loadPortfolio(baseDir) {
  const jsonPath = path.join(baseDir, 'data', 'portfolio_latest.json');
  if (!fs.existsSync(jsonPath)) return null;
  try {
    const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    return {
      updatedAt: data.updated_at || '',
      summary: data.summary || null,
      positions: data.positions || [],
    };
  } catch (e) {
    return null;
  }
}

// pipeline/update_tracking_log.py が蓄積する data/tracking_log.csv を読む。
// ポートフォリオ採用銘柄の推奨と実際の値動き(利確/損切/タイムアウト)を突き合わせた
// 記録。バックテストでは得られない、実運用でのアウトオブサンプル検証データ。
function loadTrackingLog(baseDir) {
  const csvPath = path.join(baseDir, 'data', 'tracking_log.csv');
  if (!fs.existsSync(csvPath)) return { records: [] };

  const raw = fs.readFileSync(csvPath, 'utf-8');
  const lines = raw.trim().split(/\r?\n/);
  if (lines.length <= 1) return { records: [] };

  const headers = parseCsvLine(lines[0]).map(h => h.trim());
  const records = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const row = parseCsvLine(lines[i]);
    const rec = {};
    headers.forEach((h, idx) => { rec[h] = (row[idx] || '').trim(); });
    records.push(rec);
  }
  return { records };
}

module.exports = { loadScreenedCsv, loadMacroFlowSignal, loadScreeningMeta, loadPortfolio, loadTrackingLog };