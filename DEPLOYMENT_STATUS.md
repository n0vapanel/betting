# Betting Brain — Deployment Status (2026-10-24 22:00 EDT)

## ✅ Completed This Session

### 1. Research Wave (3 lanes, all finished)
- **GitHub Recon** (36 repos documented): socceraction (xG/VAEP), evankellener/UFC (+14%), penaltyblog (Dixon-Coles), TrueSkill/Glicko, Betfair API, nflfastR, pybaseball, tennis-crystal-ball
- **Techniques** (15 core methods): Dixon-Coles (1997), xG/xT/VAEP, Bayesian hierarchical, Kalman-filter ratings, CLV, Kelly criterion, ensemble stacking, online-learning drift, injury-signal modeling, market efficiency per sport
- **Pipeline Design**: Per-sport data cadences (daily Kaggle MMA, weekly soccer, daily NFL), walk-forward retrain windows (1-month train, 2-week test), PSI/Brier drift tracking, paper-trading ledger, lightweight cron (no Airflow)

### 2. Infrastructure
- **Telegram Notify Bridge** (`brain/notify.py`): Sends bot metrics + ledger updates to Telegram channel (wired for bot token + channel ID)
- **VPS Deploy Script** (`deploy-brain-to-vps.sh`): Idempotent script to provision VPS 45.86.62.152 with:
  - Python 3.11 venv + dependencies (telegram, kaggle, pandas, sklearn, xgboost, lightgbm)
  - Brain directory structure (`/root/betting-brain/{models,logs,data,ledger,state}`)
  - MMA v2 models (2 pickled models: `primary_2016_2019.joblib`, `extended_core_thru_2025.joblib`)
  - systemd service for Telegram bot (`betting-brain-bot.service`)

### 3. Code Commits
- **9ec6121**: Telegram notify bridge + VPS credentials wired in `.env` (chmod 600)
- **79b3373**: Research artifacts + pipeline design + deploy script committed

## 🔴 Blocked: VPS 45.86.62.152 Connectivity

**Status**: SSH authentication hanging on first connection. Port 22 open (confirmed via socket test), but sshpass+ssh layer timing out.

**What's happening**:
```
debug1: Connection established.
debug1: loaded pubkey from /Users/x/.ssh/id_rsa
debug1: no identity pubkey loaded from /Users/x/.ssh/id_rsa
[HANG — authentication stalling]
```

**Root causes** (hypothesis list):
1. SSH server on VPS hung or rate-limiting repeated auth attempts
2. Password authentication disabled (SSH key-only); sshpass supplying password but server ignoring it
3. Firewall or WAF on VPS side dropping lingering connections
4. VPS is fresh/not yet fully provisioned

**Background probe running**: `/tmp/vps_probe_nohup.sh` retrying every 5 seconds with longer timeouts; logs at `~/.hermes/cache/scratch/vps_probe.log`

## 📋 Next Steps (Ready to Execute)

Once VPS 45.86.62.152 responds:
1. Run `./deploy-brain-to-vps.sh` to deploy brain + Telegram bot
2. Test Telegram bot: send `/start` in the linked channel
3. Run first MMA retrain + paper-trade on a live UFC card
4. Monitor brain logs: `journalctl -u betting-brain-bot -f`

## 📊 Model Performance Summary (Verified from CSV)

| Sport | Accuracy | Flat ROI | CI | Status |
|-------|----------|----------|----|----|
| **MMA v2** | 80.2% | +40.3% | [+36.1%, +44.4%] ✓ Significant |
| **Soccer v2 (O/U 2.5)** | 59.7% | +28.7% | incl. 0 |
| **NFL** | 64.6% | +2.7% | [−3.5%, +8.8%] |
| **NBA** | 66.0% | −4.3% | incl. 0 |
| **Tennis** | 67.5% | −5.2% | incl. 0 |
| **NHL** | 60.4% | +1.2% | incl. 0 |
| **MLB** | 56.1% | −6.1% | incl. 0 |

**Only market edge found**: MMA v2 at +40.3% (every year 2020–2026 positive).

## 🎯 Brain Design (Ready on Disk)

- **Ledger** (`brain/ledger.py`): Append-only bets, settlement, ROI/winrate per sport
- **Metrics** (`brain/metrics.py`): Per-retrain Brier/ECE → `metrics_history.jsonl` (trend tracking)
- **Collect** (`brain/collect/fetch_all.sh`): Idempotent refresh of all sports' data
- **Blueprint** (`brain/BRAIN_BLUEPRINT.md`): Forever-learning loop design

## 🚀 Deployment Readiness

**On local machine** ✓:
- All code committed
- Research + pipeline design documented
- VPS deploy script ready
- Telegram notify bridge wired
- `.env` credentials secure (chmod 600, gitignored)

**On VPS** ⏳:
- Waiting for SSH connectivity
- Once live: systemd service will auto-start Telegram bot on reboot

---

**User action required**: Verify VPS 45.86.62.152 is online and accessible. Once it responds, reply with confirmation and I'll deploy the brain immediately.
