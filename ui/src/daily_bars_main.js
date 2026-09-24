// src/daily_bars_main.js
// メインプロセス専用モジュール。daily_screening_bars_raw.parquet(build_jquants_cache.py
// が生成する縦持ちOHLCV)をhyparquetで読み、ticker別のbars配列にまとめてメモ化する。
// レンダラー側から直接 import('hyparquet') するとChromiumのESMローダーに誤って
// ルーティングされて永久にpendingのまま固まる(2026-09-24発覚)ため、確実にNode.jsとして
// 動くメインプロセス側でパースし、main.jsのIPCハンドラ(get-ticker-bars)経由で
// レンダラーに銘柄別の足データだけを返す。
const fs = require('fs');

// 1銘柄あたり保持する最大本数。このparquetは全履歴を無期限に蓄積し続けるため
// (build_jquants_cache.py側でプルーニングしていない)、上限を設けて肥大化に備える。
const MAX_BARS_PER_TICKER = 400;

let cache = null;      // Map<ticker, bar[]>
let cacheMtimeMs = null;
let cachePath = null;

async function loadDailyBarsCache(parquetPath) {
  if (!fs.existsSync(parquetPath)) {
    console.error(`[!] ${parquetPath} が見つかりません`);
    return null;
  }

  const mtimeMs = fs.statSync(parquetPath).mtimeMs;
  if (cache && cachePath === parquetPath && cacheMtimeMs === mtimeMs) {
    return cache;
  }

  const { asyncBufferFromFile, parquetReadObjects } = await import('hyparquet');
  const file = await asyncBufferFromFile(parquetPath);
  const rows = await parquetReadObjects({
    file,
    columns: ['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume'],
  });

  const byTicker = new Map();
  for (const r of rows) {
    let list = byTicker.get(r.ticker);
    if (!list) {
      list = [];
      byTicker.set(r.ticker, list);
    }
    list.push(r);
  }

  for (const [ticker, list] of byTicker) {
    list.sort((a, b) => a.Date - b.Date);
    const trimmed = list.length > MAX_BARS_PER_TICKER ? list.slice(list.length - MAX_BARS_PER_TICKER) : list;
    byTicker.set(ticker, trimmed.map(r => ({
      time: r.Date.toISOString().slice(0, 10),
      open: r.Open,
      high: r.High,
      low: r.Low,
      close: r.Close,
      volume: r.Volume,
    })));
  }

  cache = byTicker;
  cacheMtimeMs = mtimeMs;
  cachePath = parquetPath;
  console.log(`[+] daily_screening_bars_raw.parquet 読込完了 (銘柄数: ${byTicker.size}, 全${rows.length}行)`);
  return cache;
}

async function getTickerBars(parquetPath, dotCode, rawCode) {
  const c = await loadDailyBarsCache(parquetPath);
  if (!c) return null;
  return c.get(dotCode) || c.get(rawCode) || null;
}

module.exports = { getTickerBars };
