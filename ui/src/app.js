// src/app.js
const path = require('path');
const { clipboard } = require('electron');

// 修正前: require('./api') など
// 修正後: index.html起点に合わせて ./src/ を指定
const { fetchStockDetails } = require('./src/api');
const { initChart, updateChartData } = require('./src/chart');
const { loadScreenedCsv, loadMacroFlowSignal, loadScreeningMeta } = require('./src/data');
const { runFullPipeline } = require('./src/runner');

// プロジェクトルート（F:\stockModel）へのパス
const BASE_DIR = path.resolve(__dirname, '..');

let rawRecords = [];
let filteredRecords = [];
let selectedIndex = -1;
let selectedItem = null;
let currentCalc = null;

function logTerm(text, type = 'normal') {
  const term = document.getElementById('term-body');
  let prefix = '';
  if (type === 'error') prefix = '<span class="term-error">[ERROR] </span>';
  else if (type === 'info') prefix = '<span class="term-info">[INFO] </span>';
  else if (type === 'success') prefix = '<span class="term-success">[DONE] </span>';

  term.innerHTML += `\n${prefix}${text}`;
  term.scrollTop = term.scrollHeight;
}

// ヘッダーの地合いバッジ: run_dynamic_regime_screening_v8.py が実際にK(採用銘柄数)と
// サイズ倍率を決めるのに使っている地合い危険度モデル(regime_risk)の出力をそのまま表示する。
// 以前はCSVから独自ヒューリスティック(STRONG BUY比率等)で判定していたが、実際の
// スクリーニング挙動と乖離するため廃止し、モデルの生スコアに同期させた。
function updateMacroRegimeBadge() {
  const badge = document.getElementById('macro-regime-badge');
  const meta = loadScreeningMeta(BASE_DIR);
  const regimeRisk = meta && meta.regimeRisk;

  if (!regimeRisk) {
    badge.className = 'regime-badge regime-neutral';
    badge.innerText = '● 地合い危険度: データなし';
    return;
  }

  const scoreStr = (parseFloat(regimeRisk.score) || 0).toFixed(2);
  const sizeStr = `×${(parseFloat(regimeRisk.size_mult) || 1.0).toFixed(2)}`;
  const label = `● 地合い危険度 ${regimeRisk.zone} (score=${scoreStr}) K=${regimeRisk.k} サイズ${sizeStr}`;

  if (regimeRisk.zone === 'SAFE') {
    badge.className = 'regime-badge regime-bull';
  } else if (regimeRisk.zone === 'DANGER') {
    badge.className = 'regime-badge regime-bear';
  } else {
    badge.className = 'regime-badge regime-neutral';
  }
  badge.innerText = label;
  badge.title = meta.updatedAt ? `更新: ${meta.updatedAt}` : '';
}

// 空売り機会モデル(short_edge)バッジ: 「翌日TOPIXが下落する確率」の日次集約モデル。
// PESSIMISTIC(市場悲観)判定の日はSTRONG BUYでも勝率が73%→44%まで落ちることを
// バックテストで確認済みで、ロングのsize_factorを事後的に縮小(0.5倍)するのに使っている
// (地合い危険度モデルと同じ発想・同じ後処理パターン。空売り実行自体はしない)。
function updateShortEdgeBadge() {
  const badge = document.getElementById('short-edge-badge');
  if (!badge) return;
  const meta = loadScreeningMeta(BASE_DIR);
  const shortEdge = meta && meta.shortEdge;

  if (!shortEdge || shortEdge.score === null || shortEdge.score === undefined) {
    badge.className = 'regime-badge regime-neutral';
    badge.innerText = '📉 空売り機会: データなし';
    return;
  }

  const scoreStr = (parseFloat(shortEdge.score) || 0).toFixed(2);
  const sizeStr = `×${(parseFloat(shortEdge.size_mult) || 1.0).toFixed(2)}`;
  const label = `📉 空売り機会 ${shortEdge.zone} (翌日TOPIX下落確率=${scoreStr}) ロングサイズ${sizeStr}`;

  if (shortEdge.zone === 'OPTIMISTIC') {
    badge.className = 'regime-badge regime-bull';
  } else if (shortEdge.zone === 'PESSIMISTIC') {
    badge.className = 'regime-badge regime-bear';
  } else {
    badge.className = 'regime-badge regime-neutral';
  }
  badge.innerText = label;
  badge.title = meta.updatedAt ? `更新: ${meta.updatedAt}` : '';
}

// 大口手口フロー(CTA/JNET)バッジ: バックテストの結果、サイジング判断には現在使用していない
// (no_flowで確定済み)。データ収集・表示のみ継続する参考情報として、目立たないトーンに格下げする。
function updateMacroFlowBadge() {
  const badge = document.getElementById('macro-flow-badge');
  if (!badge) return;
  const data = loadMacroFlowSignal(BASE_DIR);
  badge.className = 'regime-badge flow-badge-normal';
  badge.title = 'このシグナルは現在サイジング判断には使用していません(参考表示のみ)';
  if (!data) {
    badge.innerText = '⚡ フロー: データなし (参考)';
    return;
  }
  badge.innerText = `⚡ ${data.signal_desc} (参考)`;
}

function updatePositionSize(item) {
  if (!item) return;
  const capital = parseFloat(document.getElementById('input-capital').value) || 0;
  const riskPct = parseFloat(document.getElementById('input-risk-pct').value) || 0;
  const leverage = parseFloat(document.getElementById('input-leverage').value) || 1.0;
  const maxPositionPct = parseFloat(document.getElementById('input-max-position-pct').value) || 100;
  const sizeFactor = parseFloat(item.size_factor) || 1.0;

  const maxRisk = capital * (riskPct / 100) * sizeFactor;
  const price = parseFloat(item.price);
  const stopPrice = parseFloat(item.stop);
  const riskPerShare = Math.max(1, price - stopPrice);

  // リスクベース(損切幅から逆算)の株数、信用余力(資金×レバレッジ倍率)から
  // 買える上限株数、1銘柄あたり投資上限(資金×上限%)の3つのうち最小のものを採用する。
  // 1銘柄上限が無いと、ATRが極端に小さい銘柄1〜2つが信用余力を独占し、複数銘柄を
  // 同時に買う際に他の候補が0株になったり集中リスクでMaxDDが悪化することを
  // 実資金シミュレーションで確認済み(詳細はmodules/risk_manager.pyのコメント参照)。
  const riskBasedShares = Math.floor(maxRisk / (riskPerShare * 100)) * 100;
  const maxBuyingPower = capital * leverage;
  const leverageCapShares = Math.floor(maxBuyingPower / (price * 100)) * 100;
  const maxPosYen = capital * (maxPositionPct / 100);
  const maxPosShares = Math.floor(maxPosYen / (price * 100)) * 100;
  const isLeverageCapped = leverageCapShares < riskBasedShares;
  const isMaxPosCapped = maxPosShares < Math.min(riskBasedShares, leverageCapShares);
  let shares = Math.min(riskBasedShares, leverageCapShares, maxPosShares);

  const sharesElem = document.getElementById('calc-shares');
  const factorText = sizeFactor < 1.0 ? ` (${Math.round(sizeFactor * 100)}%打診)` : '';
  const capText = isLeverageCapped ? ' [信用余力上限]' : (isMaxPosCapped ? ' [1銘柄上限]' : '');

  if (shares <= 0) {
    sharesElem.innerText = '0 株 (リスク超過)';
    sharesElem.classList.add('sizing-alert');
  } else {
    sharesElem.innerText = `${shares.toLocaleString()} 株${factorText}${capText}`;
    sharesElem.classList.remove('sizing-alert');
  }

  const actualLoss = shares * riskPerShare;
  const totalCost = (shares * price) / 10000;
  const leverageUsagePct = maxBuyingPower > 0 ? (shares * price / maxBuyingPower) * 100 : 0;

  document.getElementById('calc-risk-per-share').innerText = `${riskPerShare.toFixed(1)} 円`;
  document.getElementById('calc-actual-loss').innerText = `${Math.round(actualLoss).toLocaleString()} 円`;
  document.getElementById('calc-total-cost').innerText = `${totalCost.toFixed(1)} 万円`;

  const leverageUsageElem = document.getElementById('calc-leverage-usage');
  leverageUsageElem.innerText = `${leverageUsagePct.toFixed(1)}% (上限${Math.round(maxBuyingPower / 10000).toLocaleString()}万円)`;
  leverageUsageElem.classList.toggle('sizing-alert', isLeverageCapped);

  currentCalc = {
    shares, actualLoss: Math.round(actualLoss), totalCost: totalCost.toFixed(1),
    riskPerShare: riskPerShare.toFixed(1), sizeFactor, isLeverageCapped, isMaxPosCapped, leverage
  };
}

function renderTable(records) {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  if (records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding: 40px; color:#64748b;">該当する銘柄がありません</td></tr>';
    selectedItem = null;
    return;
  }

  records.forEach((record, index) => {
    let badgeClass = record.action.includes('STRONG BUY') ? 'badge-strong' : (record.action.includes('BUY') ? 'badge-buy' : 'badge-watch');
    const evDiff = ((parseFloat(record.ev) || 0) - (parseFloat(record.ev_raw) || (parseFloat(record.ev) || 0))).toFixed(2);
    const evDiffStr = evDiff > 0 ? `<span style="color:#4ade80; font-size:10px;">(+${evDiff})</span>` : (evDiff < 0 ? `<span style="color:#f87171; font-size:10px;">(${evDiff})</span>` : '');

    const pWinAdj = (parseFloat(record.p_win) || 0) * 100;
    const winDiff = (pWinAdj - ((parseFloat(record.p_win_raw) || (parseFloat(record.p_win) || 0)) * 100)).toFixed(1);
    const winDiffStr = winDiff > 0 ? `<span style="color:#4ade80; font-size:10px;">(+${winDiff}%)</span>` : (winDiff < 0 ? `<span style="color:#f87171; font-size:10px;">(${winDiff}%)</span>` : '');

    const sizeFactorVal = parseFloat(record.size_factor) || 1.0;
    const sizeBadge = sizeFactorVal < 1.0 ? `<span style="background: rgba(234, 179, 8, 0.2); color: #facc15; font-size: 9px; padding: 1px 4px; border-radius: 3px; border: 1px solid #eab308; margin-left: 4px;">${Math.round(sizeFactorVal*100)}%打診</span>` : '';

    let secAlertIcon = '';
    if (record.sector_shock) secAlertIcon = `<span class="table-sector-alert" title="ショック警戒: ${record.sector_summary}">🛑</span>`;
    else if (record.sector_advice === '打診30%抑制' || record.sector_score <= -0.3) secAlertIcon = `<span class="table-sector-alert" title="セクター軟調: ${record.sector_summary}">⚠️</span>`;
    else if (record.sector_score >= 0.4) secAlertIcon = `<span class="table-sector-alert" title="セクター好調: ${record.sector_summary}">🔥</span>`;

    const tr = document.createElement('tr');
    tr.className = 'data-row';
    tr.innerHTML = `
      <td style="text-align:left;">
        <strong>${record.ticker}</strong>${secAlertIcon}${sizeBadge}
        <div class="company-name" style="color: #94a3b8; font-size: 11px; max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${record.companyName || '...'}</div>
      </td>
      <td>${parseFloat(record.price).toLocaleString()}</td>
      <td>
        <div style="font-weight:bold; color:#38bdf8;">${record.ev}R ${evDiffStr}</div>
        <div style="font-size:10px; color:#64748b;">元: ${record.ev_raw || record.ev}R</div>
      </td>
      <td>
        <div>${pWinAdj.toFixed(1)}% ${winDiffStr}</div>
        <div style="font-size:10px; color:#64748b;">元: ${((parseFloat(record.p_win_raw) || (parseFloat(record.p_win) || 0)) * 100).toFixed(1)}%</div>
      </td>
      <td style="color:#94a3b8;">${(parseFloat(record.p_stop)*100).toFixed(1)}%</td>
      <td>${record.beta}</td>
      <td>
        <div>${record.vol}x</div>
        <div style="font-size:10px; color:#64748b;">${record.days_to_clear ? record.days_to_clear + '日分' : ''}</div>
      </td>
      <td style="text-align:center;"><span class="badge ${badgeClass}">${record.action.replace(/[^A-Za-z ]/g, '').trim()}</span></td>
    `;
    tr.addEventListener('click', () => selectStockByIndex(index));
    tbody.appendChild(tr);
  });

  selectStockByIndex(0);
}

async function selectStockByIndex(idx) {
  if (idx < 0 || idx >= filteredRecords.length) return;
  selectedIndex = idx;
  const item = filteredRecords[idx];
  selectedItem = item;

  const rows = document.querySelectorAll('.data-row');
  rows.forEach(r => r.classList.remove('selected'));
  if (rows[idx]) {
    rows[idx].classList.add('selected');
    rows[idx].scrollIntoView({ block: 'nearest' });
  }

  const titleElem = document.getElementById('chart-ticker-name');
  const badgeElem = document.getElementById('chart-earnings-badge');
  const marginBadge = document.getElementById('chart-margin-badge');
  const sectorBadge = document.getElementById('chart-sector-badge');

  titleElem.innerText = `${item.companyName || item.ticker} (${item.ticker}) - ${parseFloat(item.price).toLocaleString()} 円`;
  badgeElem.innerHTML = '';

  if (item.sector_name || item.sector_summary) {
    sectorBadge.style.display = 'inline-flex';
    if (item.sector_shock) {
      sectorBadge.className = 'badge-sector badge-sector-shock';
      sectorBadge.innerText = `🛑 セクターショック [${item.sector_name}]: ${item.sector_summary}`;
    } else if (item.sector_advice === '打診30%抑制' || item.sector_score <= -0.3) {
      sectorBadge.className = 'badge-sector badge-sector-warn';
      sectorBadge.innerText = `⚠️ セクター軟調 [${item.sector_name}]: ${item.sector_summary} (30%抑制)`;
    } else if (item.sector_score >= 0.4) {
      sectorBadge.className = 'badge-sector badge-sector-bull';
      const sign = item.sector_score > 0 ? '+' : '';
      sectorBadge.innerText = `🔥 セクター好調 [${item.sector_name}] (${sign}${item.sector_score.toFixed(2)}): ${item.sector_summary}`;
    } else {
      sectorBadge.style.display = 'none';
    }
  } else {
    sectorBadge.style.display = 'none';
  }

  const ratioVal = parseFloat(item.margin_ratio);
  const buyPctVal = parseFloat(item.margin_buy_pct) || 0.0;
  const isAcc = item.is_accumulating === 'True' || item.is_accumulating === true || item.is_accumulating === 'true';
  const clearDays = item.days_to_clear ? ` | 消化${item.days_to_clear}日` : '';

  if (!isNaN(ratioVal) && item.margin_ratio !== '' && item.margin_ratio !== undefined) {
    marginBadge.style.display = 'inline-block';
    if (isAcc) {
      marginBadge.style.background = 'rgba(34, 197, 94, 0.2)';
      marginBadge.style.color = '#4ade80';
      marginBadge.style.border = '1px solid #22c55e';
      marginBadge.innerText = `📦 需給良好 (買残 ${buyPctVal}%, 倍率 ${ratioVal.toFixed(2)}倍${clearDays})`;
    } else if (ratioVal > 10.0 && parseFloat(item.days_to_clear) > 2.0) {
      marginBadge.style.background = 'rgba(239, 68, 68, 0.2)';
      marginBadge.style.color = '#f87171';
      marginBadge.style.border = '1px solid #ef4444';
      marginBadge.innerText = `⚠️ 買残過多 (倍率 ${ratioVal.toFixed(2)}倍${clearDays})`;
    } else {
      marginBadge.style.background = 'rgba(148, 163, 184, 0.15)';
      marginBadge.style.color = '#94a3b8';
      marginBadge.style.border = '1px solid #64748b';
      const sign = buyPctVal > 0 ? '+' : '';
      marginBadge.innerText = `需給: 倍率 ${ratioVal.toFixed(2)}倍 (買残 ${sign}${buyPctVal}%)${clearDays}`;
    }
  } else {
    marginBadge.style.display = 'none';
  }

  const factorText = parseFloat(item.size_factor) < 1.0 ? ` | ロット: ${Math.round(parseFloat(item.size_factor)*100)}%` : '';
  document.getElementById('chart-meta-info').innerText = 
    `EV: ${item.ev}R (元: ${item.ev_raw}R) | 消化: ${item.days_to_clear || '-'}日${factorText} | β: ${item.beta} | 5日代金: ${item.turnover}億円`;
  document.getElementById('target-indicators').style.display = 'flex';
  document.getElementById('tp-val').innerText = `${item.target} 円`;
  document.getElementById('sl-val').innerText = `${item.stop} 円`;

  const priceVal = parseFloat(item.price);
  const stopVal = parseFloat(item.stop);
  const dipPrice = Math.round(priceVal - (priceVal - stopVal) * 0.5);
  item.dipPrice = dipPrice;
  document.getElementById('dip-val').innerText = `${dipPrice.toLocaleString()} 円`;

  updatePositionSize(item);

  const overlay = document.getElementById('loading-overlay');
  overlay.style.display = 'flex';

  try {
    const details = await fetchStockDetails(item.ticker);
    if (details.companyName && !item.companyName) {
      item.companyName = details.companyName;
      if (rows[idx]) {
        const nameSpan = rows[idx].querySelector('.company-name');
        if (nameSpan) nameSpan.innerText = details.companyName;
      }
      titleElem.innerText = `${details.companyName} (${item.ticker}) - ${parseFloat(item.price).toLocaleString()} 円`;
    }

    if (details.earningsDate && details.daysUntilEarnings <= 14) {
      const badgeClass = details.daysUntilEarnings <= 5 ? 'badge-earnings badge-earnings-danger' : 'badge-earnings';
      badgeElem.innerHTML = `<span class="${badgeClass}">⚠️ 決算発表: ${details.earningsDate} (残${details.daysUntilEarnings}日)</span>`;
    }

    updateChartData(details, parseFloat(item.target), stopVal, dipPrice);
  } catch (err) {
    console.error('詳細取得エラー:', err);
  } finally {
    overlay.style.display = 'none';
  }
}

function loadCsvData() {
  const { filename, records } = loadScreenedCsv(BASE_DIR);
  if (!records.length) {
    logTerm(`CSVが見つかりません、またはデータがありません`, 'error');
    return;
  }
  const pathIndicator = document.getElementById('path-indicator');
  if (pathIndicator) pathIndicator.innerText = filename;

  rawRecords = records;
  filteredRecords = [...rawRecords];
  document.getElementById('summary-text').innerText = `表示中: ${filteredRecords.length} 銘柄`;
  updateMacroRegimeBadge();
  updateShortEdgeBadge();
  updateMacroFlowBadge();
  renderTable(filteredRecords);
  logTerm(`最新CSVをロード完了: ${rawRecords.length}件`, 'info');
}

function copySbiMemo() {
  if (!selectedItem || !currentCalc) return;
  const cleanCode = selectedItem.ticker.replace('.T', '');
  const dipPriceStr = selectedItem.dipPrice ? `${selectedItem.dipPrice.toLocaleString()}円` : '-';
  let marginInfoStr = '';
  if (selectedItem.margin_ratio && selectedItem.margin_ratio !== '') {
    const ratioVal = parseFloat(selectedItem.margin_ratio);
    const buyPctVal = parseFloat(selectedItem.margin_buy_pct) || 0.0;
    const clearDays = selectedItem.days_to_clear ? `(消化${selectedItem.days_to_clear}日)` : '';
    if (!isNaN(ratioVal)) {
      marginInfoStr = `\n需給: 倍率 ${ratioVal.toFixed(2)}倍 (買残 ${buyPctVal > 0 ? '+' : ''}${buyPctVal}%) ${clearDays}`;
    }
  }

  let secMemoStr = '';
  if (selectedItem.sector_name) {
    secMemoStr = `\nセクター: ${selectedItem.sector_name} (スコア: ${selectedItem.sector_score > 0 ? '+' : ''}${selectedItem.sector_score.toFixed(2)}) - ${selectedItem.sector_summary || '特段なし'}`;
  }

  let flowNoteStr = '';
  const fData = loadMacroFlowSignal(BASE_DIR);
  if (fData) flowNoteStr = `\nマクロ手口: ${fData.signal_desc} (${fData.badge_text})`;

  const sizeNote = currentCalc.sizeFactor < 1.0 ? ` [${Math.round(currentCalc.sizeFactor * 100)}%ロット調整]` : '';
  const leverageNote = currentCalc.isLeverageCapped ? ` [信用余力上限(${currentCalc.leverage}倍)で制限]`
    : (currentCalc.isMaxPosCapped ? ' [1銘柄上限で制限]' : '');
  const memoText =
`【SBI発注メモ】
銘柄: ${cleanCode} ${selectedItem.companyName || ''}
注文: 買成 (終値 ${parseFloat(selectedItem.price).toLocaleString()}円)
      または 押し目指値: ${dipPriceStr}
数量: ${currentCalc.shares > 0 ? currentCalc.shares.toLocaleString() : 100}株${sizeNote}${leverageNote} (概算代金: ${currentCalc.totalCost}万円)
OCO設定:
  - 利確指値 (+2ATR): ${parseFloat(selectedItem.target).toLocaleString()}円
  - 損切逆指値 (-1ATR): ${parseFloat(selectedItem.stop).toLocaleString()}円
想定最大損失: ${currentCalc.actualLoss.toLocaleString()}円 (EV: ${selectedItem.ev}R)${marginInfoStr}${secMemoStr}${flowNoteStr}
理由: ${selectedItem.size_reason || selectedItem.reason || ''}`;

  clipboard.writeText(memoText);
  showCopyFeedback('btn-copy-sbi', '✅ コピー完了！', '📋 SBIメモ (C)');
  logTerm(`SBIメモをコピー: [${cleanCode}]`, 'info');
}

function copyTickerCode() {
  if (!selectedItem) return;
  const cleanCode = selectedItem.ticker.replace('.T', '');
  clipboard.writeText(cleanCode);
  showCopyFeedback('btn-copy-code', '✅ コピー！', '📋 コード (V)');
  logTerm(`コードのみコピー: ${cleanCode}`, 'info');
}

function showCopyFeedback(btnId, activeText, defaultText) {
  const btn = document.getElementById(btnId);
  btn.innerText = activeText;
  btn.classList.add('btn-copied');
  setTimeout(() => {
    btn.innerText = defaultText;
    btn.classList.remove('btn-copied');
  }, 1500);
}

// --- ポートフォリオモーダル ---
// modules/risk_manager.py の build_portfolio() と全く同じロジックをJS側にも実装し、
// ヘッダーの資金・リスク・レバレッジ・1銘柄上限の入力値でその場で再計算する。
// (以前はdata/portfolio_latest.json = Pythonバックエンドの既定値で計算済みの静的
// ファイルを表示するだけで、UIの入力値を全く見ていなかったのが原因で「レバレッジが
// 入力値と合わない」不具合になっていた)
let currentPortfolio = null;

function buildPortfolioFromCandidates(candidates, capital, riskPct, leverage, maxPositionPct, maxPositions = 20) {
  const maxBuyingPower = capital * leverage;
  const maxPosYen = capital * (maxPositionPct / 100);
  let usedBuyingPower = 0;
  let totalRiskYen = 0;
  const positions = [];

  for (const item of candidates) {
    if (positions.length >= maxPositions) break;

    const price = parseFloat(item.price);
    const stopPrice = parseFloat(item.stop);
    const sizeFactor = parseFloat(item.size_factor) || 1.0;
    const riskPerShare = Math.max(1, price - stopPrice);

    const maxRisk = capital * (riskPct / 100) * sizeFactor;
    const riskBasedShares = Math.floor(maxRisk / (riskPerShare * 100)) * 100;

    const remaining = Math.max(0, maxBuyingPower - usedBuyingPower);
    const leverageCapShares = Math.floor(remaining / (price * 100)) * 100;

    const maxPosShares = Math.floor(maxPosYen / (price * 100)) * 100;

    const shares = Math.max(0, Math.min(riskBasedShares, leverageCapShares, maxPosShares));
    if (shares <= 0) continue;

    const cost = shares * price;
    const riskYen = shares * riskPerShare;
    usedBuyingPower += cost;
    totalRiskYen += riskYen;

    positions.push({
      ...item,
      recommended_shares: shares,
      estimated_cost_yen: Math.round(cost),
      estimated_max_loss_yen: Math.round(riskYen),
      leverage_capped: leverageCapShares < riskBasedShares,
      maxpos_capped: maxPosShares < Math.min(riskBasedShares, leverageCapShares),
    });
  }

  const summary = {
    capital, leverage, max_buying_power: maxBuyingPower,
    used_buying_power_yen: Math.round(usedBuyingPower),
    buying_power_usage_pct: maxBuyingPower > 0 ? Math.round(usedBuyingPower / maxBuyingPower * 1000) / 10 : 0,
    total_risk_yen: Math.round(totalRiskYen),
    n_positions: positions.length,
    n_candidates_evaluated: candidates.length,
  };
  return { positions, summary };
}

function openPortfolioModal() {
  const overlay = document.getElementById('portfolio-modal-overlay');
  const summaryElem = document.getElementById('portfolio-summary');
  const tbody = document.getElementById('portfolio-table-body');

  const capital = parseFloat(document.getElementById('input-capital').value) || 0;
  const riskPct = parseFloat(document.getElementById('input-risk-pct').value) || 0;
  const leverage = parseFloat(document.getElementById('input-leverage').value) || 1.0;
  const maxPositionPct = parseFloat(document.getElementById('input-max-position-pct').value) || 100;

  // rawRecordsはfinal_regime_screened_v8.csvの読み込み順=既にpriority→ev_score順にソート済み
  const actionable = rawRecords.filter(r => r.action.includes('STRONG BUY') || r.action.includes('BUY'));
  const { positions, summary } = buildPortfolioFromCandidates(actionable, capital, riskPct, leverage, maxPositionPct);
  currentPortfolio = { positions, summary, updatedAt: '' };

  if (actionable.length === 0) {
    summaryElem.innerHTML = '<span style="color:#64748b;">対象銘柄(STRONG BUY / BUY)がありません</span>';
    tbody.innerHTML = '';
    overlay.hidden = false;
    return;
  }

  const s = currentPortfolio.summary;
  summaryElem.innerHTML = `
    <div class="portfolio-summary-grid">
      <div class="portfolio-summary-item"><span class="portfolio-summary-label">採用銘柄数</span><span class="portfolio-summary-val">${s.n_positions} / ${s.n_candidates_evaluated}候補</span></div>
      <div class="portfolio-summary-item"><span class="portfolio-summary-label">信用余力使用</span><span class="portfolio-summary-val">${s.buying_power_usage_pct.toFixed(1)}%</span></div>
      <div class="portfolio-summary-item"><span class="portfolio-summary-label">使用額 / 上限</span><span class="portfolio-summary-val" style="font-size:12px;">${s.used_buying_power_yen.toLocaleString()}円 / ${Math.round(s.max_buying_power).toLocaleString()}円</span></div>
      <div class="portfolio-summary-item"><span class="portfolio-summary-label">想定最大損失合計</span><span class="portfolio-summary-val" style="color:#f87171;">${s.total_risk_yen.toLocaleString()}円</span></div>
    </div>
    <div style="margin-top:6px; color:#64748b; font-size:11px;">資金${capital.toLocaleString()}円・リスク${riskPct}%・レバレッジ${leverage}倍・1銘柄上限${maxPositionPct}%で計算(現在の入力値と連動)</div>
  `;

  if (currentPortfolio.positions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:20px; color:#64748b;">該当銘柄なし</td></tr>';
  } else {
    tbody.innerHTML = currentPortfolio.positions.map(p => `
      <tr>
        <td style="text-align:left;"><strong>${p.ticker.replace('.T', '')}</strong></td>
        <td>${parseFloat(p.price).toLocaleString()}</td>
        <td style="text-align:center;">${(p.action || '').replace(/[^A-Za-z ]/g, '').trim()}</td>
        <td>${p.recommended_shares.toLocaleString()}株</td>
        <td>${p.estimated_cost_yen.toLocaleString()}円</td>
        <td>${p.estimated_max_loss_yen.toLocaleString()}円</td>
        <td>${p.leverage_capped ? '<span class="leverage-capped-tag">上限</span>' : '-'}</td>
      </tr>
    `).join('');
  }

  overlay.hidden = false;
}

function closePortfolioModal() {
  document.getElementById('portfolio-modal-overlay').hidden = true;
}

function copyPortfolioMemo() {
  if (!currentPortfolio || !currentPortfolio.positions || currentPortfolio.positions.length === 0) return;
  const s = currentPortfolio.summary;
  const lines = currentPortfolio.positions.map(p =>
    `${p.ticker.replace('.T', '')} ${(p.action || '').replace(/[^A-Za-z ]/g, '').trim()} 買成${parseFloat(p.price).toLocaleString()}円 ${p.recommended_shares.toLocaleString()}株 (概算${p.estimated_cost_yen.toLocaleString()}円${p.leverage_capped ? ' ※信用上限' : ''})`
  );
  const memoText =
`【本日のポートフォリオ発注メモ】
採用${s.n_positions}銘柄 / 信用余力使用${s.buying_power_usage_pct.toFixed(1)}% / 想定最大損失合計${s.total_risk_yen.toLocaleString()}円
--------------------------------
${lines.join('\n')}`;
  clipboard.writeText(memoText);
  showCopyFeedback('btn-copy-portfolio-memo', '✅ コピー完了！', '📋 一括SBI発注メモをコピー');
  logTerm(`ポートフォリオメモをコピー: ${s.n_positions}銘柄`, 'info');
}

document.getElementById('btn-show-portfolio').addEventListener('click', openPortfolioModal);
document.getElementById('btn-close-portfolio').addEventListener('click', closePortfolioModal);
document.getElementById('btn-copy-portfolio-memo').addEventListener('click', copyPortfolioMemo);
document.getElementById('portfolio-modal-overlay').addEventListener('click', (e) => {
  if (e.target.id === 'portfolio-modal-overlay') closePortfolioModal();
});

// イベントリスナー
document.getElementById('search-input').addEventListener('input', (e) => {
  const query = e.target.value.toLowerCase().trim();
  if (!query) {
    filteredRecords = [...rawRecords];
  } else {
    filteredRecords = rawRecords.filter(r => 
      r.ticker.toLowerCase().includes(query) || 
      (r.companyName || '').toLowerCase().includes(query) || 
      (r.sector_name || '').toLowerCase().includes(query)
    );
  }
  document.getElementById('summary-text').innerText = `表示中: ${filteredRecords.length} 銘柄`;
  renderTable(filteredRecords);
});

window.addEventListener('keydown', (e) => {
  const searchInput = document.getElementById('search-input');
  if (e.key === '/' && document.activeElement !== searchInput) {
    e.preventDefault();
    searchInput.focus();
    searchInput.select();
    return;
  }
  if (e.key === 'Escape' && document.activeElement === searchInput) {
    searchInput.blur();
    return;
  }
  if (e.key === 'Escape' && !document.getElementById('portfolio-modal-overlay').hidden) {
    closePortfolioModal();
    return;
  }
  if (document.activeElement.tagName === 'INPUT') return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (selectedIndex < filteredRecords.length - 1) selectStockByIndex(selectedIndex + 1);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (selectedIndex > 0) selectStockByIndex(selectedIndex - 1);
  } else if (e.key === 'c' || e.key === 'C') {
    e.preventDefault();
    copySbiMemo();
  } else if (e.key === 'v' || e.key === 'V') {
    e.preventDefault();
    copyTickerCode();
  }
});

document.getElementById('btn-copy-sbi').addEventListener('click', copySbiMemo);
document.getElementById('btn-copy-code').addEventListener('click', copyTickerCode);
function refreshRiskDependentViews() {
  updatePositionSize(selectedItem);
  // ポートフォリオモーダルを開いたまま資金設定を変えた場合もその場で再計算する
  if (!document.getElementById('portfolio-modal-overlay').hidden) {
    openPortfolioModal();
  }
}
document.getElementById('input-capital').addEventListener('input', refreshRiskDependentViews);
document.getElementById('input-risk-pct').addEventListener('input', refreshRiskDependentViews);
document.getElementById('input-leverage').addEventListener('input', refreshRiskDependentViews);
document.getElementById('input-max-position-pct').addEventListener('input', refreshRiskDependentViews);
document.getElementById('load-btn').addEventListener('click', loadCsvData);

document.getElementById('btn-run-all').addEventListener('click', async () => {
  const btn = document.getElementById('btn-run-all');
  const status = document.getElementById('status-indicator');
  btn.disabled = true;
  status.innerText = 'RUNNING';
  status.style.color = '#38bdf8';

  try {
    await runFullPipeline(BASE_DIR, logTerm);
    loadCsvData();
    status.innerText = 'COMPLETED';
    status.style.color = '#4ade80';
  } catch (err) {
    status.innerText = 'ERROR';
    status.style.color = '#f87171';
  } finally {
    btn.disabled = false;
  }
});

window.onload = () => {
  initChart(document.getElementById('chart-container'));
  loadCsvData();
  updateMacroFlowBadge();
};