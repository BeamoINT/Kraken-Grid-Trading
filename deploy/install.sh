#!/bin/bash
#
# Install systemd service for Kraken Grid Trading Bot
#
# Usage:
#   sudo ./deploy/install.sh
#
# After installation:
#   sudo systemctl start kraken-bot     # Start the bot
#   sudo systemctl stop kraken-bot      # Stop the bot
#   sudo systemctl restart kraken-bot   # Restart the bot
#   sudo systemctl status kraken-bot    # Check status
#   journalctl -u kraken-bot -f         # View logs
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Kraken Grid Trading Bot - Service Installation${NC}"
echo "================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo ./install.sh)${NC}"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Service file paths
SERVICE_FILE="$SCRIPT_DIR/kraken-bot.service"
SYSTEMD_DIR="/etc/systemd/system"

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}Error: Service file not found at $SERVICE_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Installing service file...${NC}"

# Copy service file
cp "$SERVICE_FILE" "$SYSTEMD_DIR/kraken-bot.service"
echo "  Copied service file to $SYSTEMD_DIR/"

# Set correct permissions
chmod 644 "$SYSTEMD_DIR/kraken-bot.service"
echo "  Set permissions to 644"

# Reload systemd daemon
echo -e "${YELLOW}Reloading systemd...${NC}"
systemctl daemon-reload
echo "  Systemd daemon reloaded"

# Enable service (start on boot)
echo -e "${YELLOW}Enabling service...${NC}"
systemctl enable kraken-bot
echo "  Service enabled (will start on boot)"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Available commands:"
echo "  sudo systemctl start kraken-bot     # Start the bot"
echo "  sudo systemctl stop kraken-bot      # Stop the bot (graceful)"
echo "  sudo systemctl restart kraken-bot   # Restart the bot"
echo "  sudo systemctl status kraken-bot    # Check status"
echo "  journalctl -u kraken-bot -f         # View logs"
echo ""
echo -e "${YELLOW}Note: The bot starts in PAPER TRADING mode by default.${NC}"
echo "To enable live trading, edit /etc/systemd/system/kraken-bot.service"
echo "and remove the --paper flag from ExecStart."
echo ""
echo "Before starting, ensure:"
echo "  1. API credentials are set in ~/kraken-grid-trading/.env"
echo "  2. Configuration is correct in ~/kraken-grid-trading/config/config.yaml"
echo ""
