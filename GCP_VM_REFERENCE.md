# GCP VM Reference - Kraken Grid Trading Bot

This document contains all information needed to SSH into and manage the Google Cloud VM running the Kraken Grid Trading Bot.

---

## Quick Reference

```bash
# SSH into VM (simplest)
ssh kraken-bot

# Check bot status
ssh kraken-bot "sudo systemctl status kraken-bot"

# View bot logs
ssh kraken-bot "journalctl -u kraken-bot -f"

# Deploy latest code
ssh kraken-bot "cd ~/kraken-grid-trading && git pull origin main"
```

---

## SSH Configuration

### Connection Details

| Property | Value |
|----------|-------|
| **VM Name** | krakengridbot |
| **External IP** | 35.227.103.77 |
| **Username** | beamo_beamosupport_com |
| **SSH Key (local)** | ~/.ssh/kraken_gcp |
| **SSH Config Aliases** | kraken-bot, gcp-kraken |
| **GCP Zone** | (check GCP Console) |
| **GCP Project** | (check GCP Console) |

### SSH Commands

```bash
# Using SSH config alias (recommended)
ssh kraken-bot

# Alternative alias
ssh gcp-kraken

# Direct command (if aliases not configured)
ssh -i ~/.ssh/kraken_gcp beamo_beamosupport_com@35.227.103.77

# With verbose output (for debugging)
ssh -v kraken-bot

# Copy file to VM
scp local_file.txt kraken-bot:~/kraken-grid-trading/

# Copy file from VM
scp kraken-bot:~/kraken-grid-trading/remote_file.txt ./
```

### SSH Config File (~/.ssh/config)

Add this to your local `~/.ssh/config` file for easy access:

```
Host kraken-bot gcp-kraken
    HostName 35.227.103.77
    User beamo_beamosupport_com
    IdentityFile ~/.ssh/kraken_gcp
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### SSH Key Setup (if needed)

```bash
# Generate new key (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/kraken_gcp -C "kraken-gcp-vm"

# View public key (add this to GCP VM metadata or authorized_keys)
cat ~/.ssh/kraken_gcp.pub

# Set correct permissions
chmod 600 ~/.ssh/kraken_gcp
chmod 644 ~/.ssh/kraken_gcp.pub
```

---

## VM System Specifications

| Resource | Value |
|----------|-------|
| **OS** | Ubuntu 24.04.3 LTS (noble) |
| **Kernel** | 6.14.0-1021-gcp |
| **Architecture** | x86_64 |
| **CPU** | Intel Xeon @ 2.20GHz |
| **CPU Cores** | 2 |
| **RAM** | 3.8 GB |
| **Disk** | 43 GB (41 GB available) |
| **Python** | 3.12.3 |
| **Machine Type** | e2-small or similar |

---

## File Paths on VM

| Path | Purpose |
|------|---------|
| `/home/beamo_beamosupport_com` | Home directory (~) |
| `~/kraken-grid-trading` | Bot installation directory |
| `~/kraken-grid-trading/venv` | Python virtual environment |
| `~/kraken-grid-trading/.env` | API credentials (NEVER commit!) |
| `~/kraken-grid-trading/config/config.yaml` | Bot configuration |
| `~/kraken-grid-trading/data/` | Database and state files |
| `~/kraken-grid-trading/logs/` | Log files |
| `~/kraken-grid-trading/data/trading.db` | SQLite database |
| `~/kraken-grid-trading/data/trading.pid` | PID lock file |

---

## Systemd Service Management

The bot runs as a systemd service called `kraken-bot`.

### Service Commands

```bash
# Start the bot
ssh kraken-bot "sudo systemctl start kraken-bot"

# Stop the bot (graceful shutdown)
ssh kraken-bot "sudo systemctl stop kraken-bot"

# Restart the bot
ssh kraken-bot "sudo systemctl restart kraken-bot"

# Check status
ssh kraken-bot "sudo systemctl status kraken-bot"

# View logs (live)
ssh kraken-bot "journalctl -u kraken-bot -f"

# View last 100 log lines
ssh kraken-bot "journalctl -u kraken-bot -n 100"

# View logs since last boot
ssh kraken-bot "journalctl -u kraken-bot -b"

# Enable auto-start on boot
ssh kraken-bot "sudo systemctl enable kraken-bot"

# Disable auto-start
ssh kraken-bot "sudo systemctl disable kraken-bot"
```

### Service Configuration

The service file is at `/etc/systemd/system/kraken-bot.service`.

**Default mode**: Paper trading (`--paper` flag)

To switch to live trading:
```bash
# Edit service file
ssh kraken-bot "sudo nano /etc/systemd/system/kraken-bot.service"
# Remove --paper from ExecStart line, then:
ssh kraken-bot "sudo systemctl daemon-reload && sudo systemctl restart kraken-bot"
```

---

## Common Operations

### Deploy Code Updates

```bash
# Full deployment sequence
ssh kraken-bot "cd ~/kraken-grid-trading && git pull origin main"
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && pip install -r requirements.txt"
ssh kraken-bot "sudo systemctl restart kraken-bot"
```

### Quick Deploy (code only, no deps)

```bash
ssh kraken-bot "cd ~/kraken-grid-trading && git pull origin main && sudo systemctl restart kraken-bot"
```

### Validate Configuration

```bash
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && python -m src.main config/config.yaml --dry-run"
```

### Run Bot Manually (for testing)

```bash
# Paper trading mode
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && python -m src.main config/config.yaml --paper"

# Fresh start (clear state)
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && python -m src.main config/config.yaml --paper --fresh"

# With debug logging
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && python -m src.main config/config.yaml --paper --log-level DEBUG"
```

### View Logs

```bash
# Systemd journal (recommended when running as service)
ssh kraken-bot "journalctl -u kraken-bot -f"

# Application log file
ssh kraken-bot "tail -f ~/kraken-grid-trading/logs/trading.log"

# Last 50 lines of log
ssh kraken-bot "tail -50 ~/kraken-grid-trading/logs/trading.log"
```

### Check System Resources

```bash
# Disk space
ssh kraken-bot "df -h"

# Memory usage
ssh kraken-bot "free -h"

# CPU and processes
ssh kraken-bot "htop"  # or "top"

# Check if bot process is running
ssh kraken-bot "ps aux | grep python"

# Check database size
ssh kraken-bot "ls -lh ~/kraken-grid-trading/data/"
```

### Manage Environment Variables

```bash
# View current .env (CAREFUL - contains secrets)
ssh kraken-bot "cat ~/kraken-grid-trading/.env"

# Edit .env file
ssh kraken-bot "nano ~/kraken-grid-trading/.env"

# Required variables:
# KRAKEN_API_KEY=your_api_key_here
# KRAKEN_API_SECRET=your_api_secret_here
# PAPER_TRADING=true
```

### Run Tests

```bash
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && python -m pytest tests/ -v"
```

### Update System Packages

```bash
ssh kraken-bot "sudo apt update && sudo apt upgrade -y"
```

---

## GCP Console Access

- **GCP Console**: https://console.cloud.google.com/
- **VM Instances**: Compute Engine → VM instances → krakengridbot
- **SSH via Browser**: Click "SSH" button on VM instance page (backup method)
- **Serial Console**: For debugging boot issues
- **Firewall Rules**: VPC Network → Firewall (if needed)

---

## Initial Setup Checklist

For setting up the VM from scratch:

- [x] VM created in GCP
- [x] SSH key configured
- [x] SSH connection verified
- [x] Git installed
- [x] Python 3.12 installed
- [x] Repository cloned (`git clone https://github.com/BeamoINT/Kraken-Grid-Trading.git ~/kraken-grid-trading`)
- [x] Virtual environment created (`python3 -m venv ~/kraken-grid-trading/venv`)
- [x] Dependencies installed (`pip install -r requirements.txt`)
- [x] .env file created from example
- [x] Config file configured
- [x] Dry-run test passed
- [x] Systemd service installed (`sudo ./deploy/install.sh`)
- [ ] API credentials added to .env
- [ ] Paper trading tested
- [ ] Live trading enabled (when ready)

---

## CLI Arguments Reference

```bash
python -m src.main config/config.yaml [OPTIONS]

Options:
  --paper           Force paper trading mode (orders validated but not executed)
  --fresh           Clear saved state and start fresh
  --dry-run         Validate configuration and exit without trading
  --log-level LEVEL Set logging level (DEBUG, INFO, WARNING, ERROR)
  --log-file PATH   Log file path
  --force-start     Override PID lock (use if previous instance crashed)
  --no-resume       Skip state restoration (but keep history)
  --max-state-age N Maximum state age in seconds for resume (default: 86400)
```

---

## Troubleshooting

### SSH Connection Issues

```bash
# Test with verbose output
ssh -v kraken-bot

# Check if VM is running (GCP Console → Compute Engine → VM instances)

# If permission denied, check authorized_keys on VM (via GCP Console SSH):
cat ~/.ssh/authorized_keys

# Verify local SSH key permissions
ls -la ~/.ssh/kraken_gcp*
# Should be: -rw------- for private key, -rw-r--r-- for public key
```

### Bot Won't Start

```bash
# Check service status
ssh kraken-bot "sudo systemctl status kraken-bot"

# Check for PID lock file (from crashed instance)
ssh kraken-bot "cat ~/kraken-grid-trading/data/trading.pid"
ssh kraken-bot "rm ~/kraken-grid-trading/data/trading.pid"  # Remove if stale

# Check configuration is valid
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && python -m src.main config/config.yaml --dry-run"

# Check .env file exists and has credentials
ssh kraken-bot "ls -la ~/kraken-grid-trading/.env"
```

### VM Not Responding

1. Check VM status in GCP Console
2. Try stopping and starting the VM (note: IP may change!)
3. Check serial console output in GCP Console
4. If IP changed, update SSH config

### Database Issues

```bash
# Check database file
ssh kraken-bot "ls -la ~/kraken-grid-trading/data/trading.db"

# Check database stats
ssh kraken-bot "cd ~/kraken-grid-trading && source venv/bin/activate && python -c \"from src.core import StateManager; sm = StateManager(); print(sm.get_stats())\""
```

---

## Security Notes

- **NEVER** commit `.env` file or API credentials to git
- SSH key location: `~/.ssh/kraken_gcp` (private), `~/.ssh/kraken_gcp.pub` (public)
- Keep SSH private key secure (permissions 600)
- Use paper trading mode until you're confident in the setup
- Consider setting up GCP firewall rules to restrict access
- The VM's external IP may change if stopped/started without a static IP

---

## Useful One-Liners

```bash
# Quick status check
ssh kraken-bot "sudo systemctl status kraken-bot --no-pager && echo '---' && df -h / && echo '---' && free -h"

# Tail logs and restart
ssh kraken-bot "sudo systemctl restart kraken-bot && journalctl -u kraken-bot -f"

# Full redeploy
ssh kraken-bot "cd ~/kraken-grid-trading && git pull && source venv/bin/activate && pip install -r requirements.txt && sudo systemctl restart kraken-bot && journalctl -u kraken-bot -f"

# Check if bot is actively trading
ssh kraken-bot "journalctl -u kraken-bot -n 20 --no-pager"

# Emergency stop
ssh kraken-bot "sudo systemctl stop kraken-bot"
```

---

*Last updated: 2026-01-28*

*Note: External IP (35.227.103.77) may change if VM is stopped/started. Consider reserving a static IP in GCP if this becomes an issue.*
