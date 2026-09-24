// src/api.js
const fs = require('fs');
const path = require('path');
const { ipcRenderer } = require('electron');

// プロジェクトの絶対パスを直接指定してパスのズレを完全に排除
const BASE_DIR = path.resolve(__dirname, '..', '..'); // プロジェクトのルートディレクトリを取得
const CACHE_DIR = path.join(BASE_DIR, 'data', 'cache');
const SECTOR_MASTER_PATH = path.join(BASE_DIR, 'data', 'jpx_sector_master.json');
// build_jquants_cache.pyが縦持ち(1行=1銘柄1日)で蓄積するOHLCVキャッシュ。以前は
// これをpivotしてticker別に再整形したchart_candles_*.jsonを別途生成していたが、
// hyparquet(依存ゼロの純JSパーサ)でこのparquetを直接読めるようになったため、
// 中間JSONの生成・維持(2箇所を同期させる手間。分割調整バグ修復時に両方個別に
// 直す必要があった)をやめた(2026-09-24)。
// ただしhyparquetのパース自体はレンダラーではなくメインプロセス側(src/daily_bars_main.js)
// で行う。レンダラー(nodeIntegration:trueのclassicスクリプト)からimport('hyparquet')を
// 直接呼ぶと、Node.jsの動的importではなくChromiumページ側のESMローダーに誤って
// ルーティングされ、永久にpendingのままUIごとフリーズする不具合があったため
// (「UIがまるっきり表示されない」で発覚)、main.jsのIPC(get-ticker-bars)経由で取得する。

let sectorMaster = null;
const datedCacheMemo = {}; // prefix -> { path, data }

function getLocalTodayStr() {
  const d = new Date();
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
}

// data/cache/ 配下の "<prefix>_YYYYMMDD.json" 形式のキャッシュを読み込む。
// 本日分が無ければ直近の日付のファイルにフォールバックする(build_jquants_cache.py
// が生成するchart_candles/company_master/earnings_calendarで共通利用)。
function loadLatestDatedJson(prefix) {
  if (!fs.existsSync(CACHE_DIR)) {
    console.error(`[!] キャッシュフォルダが存在しません: ${CACHE_DIR}`);
    return null;
  }

  const memo = datedCacheMemo[prefix];
  const todayFile = path.join(CACHE_DIR, `${prefix}_${getLocalTodayStr()}.json`);

  if (fs.existsSync(todayFile)) {
    if (memo && memo.path === todayFile) return memo.data;
    try {
      const data = JSON.parse(fs.readFileSync(todayFile, 'utf-8'));
      datedCacheMemo[prefix] = { path: todayFile, data };
      console.log(`[+] ${prefix}: 本日分キャッシュ読込成功 (登録数: ${Object.keys(data).length})`);
      return data;
    } catch (e) {
      console.error(`[!] ${prefix}: 当日JSONパースエラー:`, e);
    }
  }

  try {
    const files = fs.readdirSync(CACHE_DIR)
      .filter(f => f.startsWith(`${prefix}_`) && f.endsWith('.json'))
      .sort()
      .reverse();

    if (files.length > 0) {
      const latestFile = path.join(CACHE_DIR, files[0]);
      if (memo && memo.path === latestFile) return memo.data;
      const data = JSON.parse(fs.readFileSync(latestFile, 'utf-8'));
      datedCacheMemo[prefix] = { path: latestFile, data };
      console.log(`[+] ${prefix}: 直近ファイルから読込成功 (${files[0]}, 登録数: ${Object.keys(data).length})`);
      return data;
    }
    console.warn(`[!] ${CACHE_DIR} 内に ${prefix}_*.json が1つもありません。`);
  } catch (e) {
    console.error(`[!] ${prefix}: フォルダ走査エラー:`, e);
  }

  return null;
}

function loadSectorMaster() {
  if (sectorMaster) return sectorMaster;
  sectorMaster = {};
  if (!fs.existsSync(SECTOR_MASTER_PATH)) {
    console.warn(`[!] 銘柄マスターが見つかりません: ${SECTOR_MASTER_PATH}`);
    return sectorMaster;
  }
  try {
    sectorMaster = JSON.parse(fs.readFileSync(SECTOR_MASTER_PATH, 'utf-8'));
  } catch (e) {
    console.error('[!] 銘柄マスター読込エラー:', e);
    sectorMaster = {};
  }
  return sectorMaster;
}

function getCompanyName(rawCode, dotCode) {
  // 1. 上場銘柄マスター(全銘柄、ETF/REIT含む、build_jquants_cache.py PART6)
  const companyMaster = loadLatestDatedJson('company_master') || {};
  if (companyMaster[dotCode] && companyMaster[dotCode].name) {
    return companyMaster[dotCode].name;
  }
  // 2. セクターセンチメント用マスター(株式のみ)へのフォールバック
  const sector = loadSectorMaster();
  if (sector[rawCode] && sector[rawCode].name) {
    return sector[rawCode].name;
  }
  return rawCode;
}

function getEarningsInfo(dotCode) {
  const calendar = loadLatestDatedJson('earnings_calendar') || {};
  const info = calendar[dotCode];
  if (!info || !info.next_earnings_date) {
    return { earningsDate: null, daysUntilEarnings: 999 };
  }
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const target = new Date(info.next_earnings_date);
  const daysUntilEarnings = Math.round((target - today) / (1000 * 60 * 60 * 24));
  return { earningsDate: info.next_earnings_date, daysUntilEarnings };
}

async function fetchStockDetails(ticker) {
  const rawCode = ticker.replace('.T', '').trim();
  const dotCode = `${rawCode}.T`;
  const companyName = getCompanyName(rawCode, dotCode);
  const { earningsDate, daysUntilEarnings } = getEarningsInfo(dotCode);

  const rawBars = await ipcRenderer.invoke('get-ticker-bars', dotCode, rawCode);

  if (!rawBars || rawBars.length === 0) {
    console.warn(`[!] 銘柄 [${ticker}] の足データがキャッシュ内にありません（キー候補: ${dotCode}, ${rawCode}）`);
    return { companyName, bars: [], volumes: [], earningsDate, daysUntilEarnings };
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
    companyName,
    bars,
    volumes,
    earningsDate,
    daysUntilEarnings
  };
}

module.exports = { fetchStockDetails };