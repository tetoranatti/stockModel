// src/chart.js
const LightweightCharts = window.LightweightCharts;

let currentChart = null;
let candleSeries = null;
let volumeSeries = null;
let sma25Series = null;
let sma75Series = null;
let tpLine = null;
let slLine = null;
let dipLine = null;

function calculateSMA(data, period) {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue;
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close;
    }
    result.push({ time: data[i].time, value: sum / period });
  }
  return result;
}

function initChart(containerElem) {
  containerElem.innerHTML = '<div class="loading-overlay" id="loading-overlay">チャートデータ取得中...</div>';

  currentChart = LightweightCharts.createChart(containerElem, {
    width: containerElem.clientWidth || 600,
    height: containerElem.clientHeight || 400,
    layout: {
      background: { color: '#0b0f19' },
      textColor: '#94a3b8',
      panes: {
        separatorColor: '#334155',
        separatorHoverColor: '#38bdf8',
        enableResize: true,
      },
    },
    grid: {
      vertLines: { color: '#1e293b' },
      horzLines: { color: '#1e293b' },
    },
    timeScale: {
      borderColor: '#334155',
      timeVisible: true,
      rightOffset: 8,
      barSpacing: 6,
    },
    rightPriceScale: {
      borderColor: '#334155',
      autoScale: true,
      scaleMargins: { top: 0.1, bottom: 0.1 },
    }
  });

  // Pane 0: 上部株価ペイン
  const candleOpts = {
    upColor: '#ef4444',
    downColor: '#3b82f6',
    borderUpColor: '#ef4444',
    borderDownColor: '#3b82f6',
    wickUpColor: '#ef4444',
    wickDownColor: '#3b82f6',
    priceScaleId: 'right',
  };
  candleSeries = currentChart.addSeries(LightweightCharts.CandlestickSeries, candleOpts, 0);

  const sma25Opts = { color: '#f59e0b', lineWidth: 2, crosshairMarkerVisible: false, priceScaleId: 'right' };
  const sma75Opts = { color: '#a855f7', lineWidth: 2, crosshairMarkerVisible: false, priceScaleId: 'right' };
  sma25Series = currentChart.addSeries(LightweightCharts.LineSeries, sma25Opts, 0);
  sma75Series = currentChart.addSeries(LightweightCharts.LineSeries, sma75Opts, 0);

  // Pane 1: 下部独立出来高ペイン
  const volOpts = { priceFormat: { type: 'volume' }, priceScaleId: 'right' };
  volumeSeries = currentChart.addSeries(LightweightCharts.HistogramSeries, volOpts, 1);

  if (typeof currentChart.panes === 'function') {
    const panes = currentChart.panes();
    if (panes && panes[1]) panes[1].setHeight(100);
  }

  window.addEventListener('resize', () => {
    if (currentChart && containerElem) {
      currentChart.resize(containerElem.clientWidth, containerElem.clientHeight);
    }
  });
}

function updateChartData(details, targetPrice, stopPrice, dipPrice) {
  if (!candleSeries) return;

  candleSeries.setData(details.bars);

  if (volumeSeries && details.volumes.length > 0) {
    volumeSeries.setData(details.volumes);
  }
  if (sma25Series && details.bars.length >= 25) {
    sma25Series.setData(calculateSMA(details.bars, 25));
  }
  if (sma75Series && details.bars.length >= 75) {
    sma75Series.setData(calculateSMA(details.bars, 75));
  }

  // 水平ライン更新
  if (tpLine) candleSeries.removePriceLine(tpLine);
  if (slLine) candleSeries.removePriceLine(slLine);
  if (dipLine) candleSeries.removePriceLine(dipLine);

  tpLine = candleSeries.createPriceLine({
    price: targetPrice,
    color: '#22c55e',
    lineWidth: 2,
    lineStyle: 0,
    axisLabelVisible: true,
    title: '利確 (+2.0 ATR)',
  });

  dipLine = candleSeries.createPriceLine({
    price: dipPrice,
    color: '#38bdf8',
    lineWidth: 2,
    lineStyle: 2,
    axisLabelVisible: true,
    title: '押し目待機 (-0.5 ATR)',
  });

  slLine = candleSeries.createPriceLine({
    price: stopPrice,
    color: '#ef4444',
    lineWidth: 2,
    lineStyle: 0,
    axisLabelVisible: true,
    title: '損切 (-1.0 ATR)',
  });

  // 直近70営業日フォーカス
  if (details.bars.length > 0) {
    const totalBars = details.bars.length;
    const fromIdx = Math.max(0, totalBars - 70);
    currentChart.timeScale().setVisibleRange({
      from: details.bars[fromIdx].time,
      to: details.bars[totalBars - 1].time
    });
  }
}

module.exports = { initChart, updateChartData };