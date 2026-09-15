// src/api.js
const fs = require('fs');
const path = require('path');

// プロジェクトの絶対パスを直接指定してパスのズレを完全に排除
const BASE_DIR = path.resolve(__dirname, '..', '..'); // プロジェクトのルートディレクトリを取得
const CACHE_DIR = path.join(BASE_DIR, 'data', 'cache');

let loadedChartCache = null;
let loadedCachePath = '';

function getLocalTodayStr() {
  const d = new Date();
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
}

function getLatestChartCache() {
  // 1. キャッシュフォルダの存在確認
  if (!fs.existsSync(CACHE_DIR)) {
    console.error(`[!] キャッシュフォルダが存在しません: ${CACHE_DIR}`);
    return null;
  }

  const todayStr = getLocalTodayStr();
  const todayFile = path.join(CACHE_DIR, `chart_candles_${todayStr}.json`);

  // 2. 本日分が存在する場合
  if (fs.existsSync(todayFile)) {
    if (loadedCachePath === todayFile && loadedChartCache) {
      return loadedChartCache;
    }
    try {
      console.log(`[*] 本日分キャッシュをロード: ${todayFile}`);
      const content = fs.readFileSync(todayFile, 'utf-8');
      loadedChartCache = JSON.parse(content);
      loadedCachePath = todayFile;
      console.log(`[+] キャッシュ読込成功 (登録銘柄数: ${Object.keys(loadedChartCache).length})`);
      return loadedChartCache;
    } catch (e) {
      console.error('[!] 当日JSONパースエラー:', e);
    }
  }

  // 3. 本日分がない場合、最新の chart_candles_*.json を探す
  try {
    const files = fs.readdirSync(CACHE_DIR)
      .filter(f => f.startsWith('chart_candles_') && f.endsWith('.json'))
      .sort()
      .reverse();

    if (files.length > 0) {
      const latestFile = path.join(CACHE_DIR, files[0]);
      if (loadedCachePath === latestFile && loadedChartCache) {
        return loadedChartCache;
      }
      console.log(`[*] 直近ファイルからロード: ${latestFile}`);
      const content = fs.readFileSync(latestFile, 'utf-8');
      loadedChartCache = JSON.parse(content);
      loadedCachePath = latestFile;
      console.log(`[+] キャッシュ読込成功 (登録銘柄数: ${Object.keys(loadedChartCache).length})`);
      return loadedChartCache;
    } else {
      console.warn(`[!] ${CACHE_DIR} 内に chart_candles_*.json が1つもありません。`);
    }
  } catch (e) {
    console.error('[!] フォルダ走査エラー:', e);
  }

  return null;
}

async function fetchStockDetails(ticker) {
  const rawCode = ticker.replace('.T', '').trim();
  const dotCode = `${rawCode}.T`;

  const cache = getLatestChartCache();

  if (!cache) {
    console.error(`[!] キャッシュオブジェクトが空です (CACHE_DIR: ${CACHE_DIR})`);
    return { companyName: rawCode, bars: [], volumes: [] };
  }

  const rawBars = cache[dotCode] || cache[rawCode];

  if (!rawBars || rawBars.length === 0) {
    console.warn(`[!] 銘柄 [${ticker}] の足データがJSON内にありません（キー候補: ${dotCode}, ${rawCode}）`);
    return { companyName: rawCode, bars: [], volumes: [] };
  }

  const bars = [];
  const volumes = [];

  for (const b of rawBars) {
    bars.push({
      time: b.time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close
    });

    volumes.push({
      time: b.time,
      value: b.volume,
      color: b.close >= b.open ? 'rgba(239, 68, 68, 0.4)' : 'rgba(59, 130, 246, 0.4)'
    });
  }

  return {
    companyName: rawCode,
    bars,
    volumes,
    earningsDate: null,
    daysUntilEarnings: 999
  };
}

module.exports = { fetchStockDetails };