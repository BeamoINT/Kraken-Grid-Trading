#!/bin/bash
#
# Uninstall systemd service for Kraken Grid Trading Bot
#
# Usage:
#   sudo ./deploy/uninstall.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Kraken Grid Trading Bot - Service Uninstallation${NC}"
echo "=================================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo ./uninstall.sh)${NC}"
    exit 1
fi

SERVICE_FILE="/etc/systemd/system/kraken-bot.service"

# Check if service exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${YELLOW}Service is not installed.${NC}"
    exit 0
fi

# Stop service if running
echo -e "${YELLOW}Stopping service...${NC}"
systemctl stop kraken-bot 2>/dev/null || true
echo "  Service stopped"

# Disable service
echo -e "${YELLOW}Disabling service...${NC}"
systemctl disable kraken-bot 2>/dev/null || true
echo "  Service disabled"

# Remove service file
echo -e "${YELLOW}Removing service file...${NC}"
rm -f "$SERVICE_FILE"
echo "  Service file removed"

# Reload systemd daemon
echo -e "${YELLOW}Reloading systemd...${NC}"
systemctl daemon-reload
echo "  Systemd daemon reloaded"

echo ""
echo -e "${GREEN}Uninstallation complete!${NC}"
echo ""
echo "Note: Bot data, configuration, and logs have NOT been removed."
echo "To fully remove, delete ~/kraken-grid-trading"
echo ""
