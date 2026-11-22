"""Report output: JSON for programs, text for the terminal, a single HTML file to share."""

from __future__ import annotations

import html
import json

import pandas as pd

from .engine import BacktestResult
from .explain import explain

_LABELS = [
    ("net_pnl", "Net P&L"),
    ("net_return_pct", "Return %"),
    ("cagr_pct", "CAGR %"),
    ("max_drawdown_pct", "Max drawdown %"),
    ("sharpe", "Sharpe"),
    ("sortino", "Sortino"),
    ("trades", "Trades"),
    ("win_rate_pct", "Win rate %"),
    ("profit_factor", "Profit factor"),
    ("expectancy", "Avg P&L per trade"),
    ("avg_win", "Avg win"),
    ("avg_loss", "Avg loss"),
    ("best_trade", "Best trade"),
    ("worst_trade", "Worst trade"),
    ("max_losing_streak", "Longest losing streak"),
    ("exposure_pct", "Time in market %"),
    ("total_costs", "Total costs"),
]


def to_dict(r: BacktestResult, max_points: int = 1500) -> dict:
    eq = r.equity
    if len(eq) > max_points:
        step = len(eq) // max_points + 1
        eq = pd.concat([eq.iloc[::step], eq.iloc[[-1]]])
        eq = eq[~eq.index.duplicated()]
    trades = r.trades.copy()
    for col in ("entry_time", "exit_time"):
        trades[col] = trades[col].astype(str)
    return {
        "strategy": r.strategy.model_dump(mode="json", exclude_none=True),
        "rules": explain(r.strategy),
        "data_source": r.data_source,
        "start": r.start,
        "end": r.end,
        "bars": r.bars,
        "metrics": r.metrics,
        "benchmark_return_pct": r.benchmark_return_pct,
        "notices": r.notices,
        "equity": [{"t": str(t), "v": round(float(v), 2)} for t, v in eq.items()],
        "trades": trades.to_dict(orient="records"),
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def to_text(r: BacktestResult) -> str:
    m = r.metrics
    out = [f"{r.strategy.name}", f"{r.data_source}, {r.start} to {r.end} ({r.bars} bars)", ""]
    width = max(len(label) for _, label in _LABELS)
    for key, label in _LABELS:
        out.append(f"  {label:<{width}}  {_fmt(m.get(key))}")
    if r.benchmark_return_pct is not None:
        label = "Buy and hold %" if r.strategy.side == "long" else "Short and hold %"
        out.append(f"  {label:<{width}}  {_fmt(r.benchmark_return_pct)}")
    if r.notices:
        out += ["", "Notes:"] + [f"  - {n['message']}" for n in r.notices]
    if len(r.trades):
        out += ["", "Last trades:"]
        for t in r.trades.tail(10).itertuples():
            out.append(
                f"  {str(t.entry_time)[:16]} -> {str(t.exit_time)[:16]}  {t.side:<5} qty {t.quantity:<6} "
                f"{t.entry_price:>10.2f} -> {t.exit_price:>10.2f}  P&L {t.pnl:>10.2f}  ({t.exit_reason})"
            )
    return "\n".join(out)


def to_html(r: BacktestResult) -> str:
    data = json.dumps(to_dict(r))
    title = html.escape(r.strategy.name)
    return _HTML.replace("__TITLE__", title).replace("__DATA__", data.replace("</", "<\\/"))


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · plainquant</title>
<style>
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--up:#127a4a;--down:#b3261e;--accent:#1f5fbf}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#9a9993;--line:#2c2c2a;--up:#3fbf7f;--down:#ef6b61;--accent:#7aa7ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:18px 0}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px}.tile b{display:block;font-size:18px}
svg{width:100%;height:240px;border:1px solid var(--line);border-radius:8px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}.wrap{overflow-x:auto}
.up{color:var(--up)}.down{color:var(--down)}
</style></head><body><main id="app"></main>
<script>
const R = __DATA__;
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f = v => v === null || v === undefined ? '-' : (typeof v === 'number' ? v.toLocaleString(undefined,{maximumFractionDigits:2}) : v);
const m = R.metrics;
const tiles = [['Net P&L',m.net_pnl],['Return %',m.net_return_pct],['Max drawdown %',m.max_drawdown_pct],['Win rate %',m.win_rate_pct],
 ['Trades',m.trades],['Profit factor',m.profit_factor],['Sharpe',m.sharpe],['Buy & hold %',R.benchmark_return_pct]];
function curve(points){
  if(points.length<2) return '';
  const v=points.map(p=>p.v), lo=Math.min(...v), hi=Math.max(...v), W=1000, H=240, pad=10;
  const x=i=>pad+i*(W-2*pad)/(v.length-1), y=a=>H-pad-(a-lo)*(H-2*pad)/((hi-lo)||1);
  const d=v.map((a,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(a).toFixed(1)).join('');
  const up=v[v.length-1]>=v[0];
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Equity curve"><path d="${d}" fill="none" stroke="var(${up?'--up':'--down'})" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
}
document.getElementById('app').innerHTML = `
<h1>${esc(R.strategy.name)}</h1>
<div class="muted">${esc(R.data_source)} · ${R.start} to ${R.end} · ${R.bars} bars</div>
<ul>${R.rules.map(r=>`<li>${esc(r)}</li>`).join('')}</ul>
<div class="grid">${tiles.map(([k,v])=>`<div class="tile"><span class="muted">${k}</span><b class="${typeof v==='number'&&k!=='Trades'?(v>=0?'up':'down'):''}">${f(v)}</b></div>`).join('')}</div>
${curve(R.equity)}
${R.notices.length?`<h3>Notes</h3><ul>${R.notices.map(n=>`<li>${esc(n.message)}</li>`).join('')}</ul>`:''}
<h3>Trades (${R.trades.length})</h3>
<div class="wrap"><table><tr><th>Entry</th><th>Exit</th><th>Qty</th><th>Entry px</th><th>Exit px</th><th>P&L</th><th>Reason</th></tr>
${R.trades.map(t=>`<tr><td>${esc(t.entry_time.slice(0,16))}</td><td>${esc(t.exit_time.slice(0,16))}</td><td>${t.quantity}</td><td>${f(t.entry_price)}</td><td>${f(t.exit_price)}</td><td class="${t.pnl>=0?'up':'down'}">${f(t.pnl)}</td><td>${esc(t.exit_reason)}</td></tr>`).join('')}
</table></div>
<p class="muted">Made with plainquant. Past results do not predict future returns. Not investment advice.</p>`;
</script></body></html>
"""
