#!/usr/bin/env bash
# setup.sh — Install the expflow reverse pipeline skill.
#
# Usage: bash setup.sh
#
# This script:
#   1. Creates ~/.hermes/task_monitor/
#   2. Symlinks taskctl.py and qq_send.py into it
#   3. Creates taskctl.conf template if not exists
#   4. Optionally adds crontab entries
#
# Run this after: hermes skills install expflow-reverse-pipeline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_DIR="$HOME/.hermes/task_monitor"

echo "[OK] Installing expflow reverse pipeline..."

# Create monitor directory
mkdir -p "$MONITOR_DIR"

# Symlink scripts
for script in taskctl.py qq_send.py; do
    src="$SCRIPT_DIR/$script"
    dst="$MONITOR_DIR/$script"
    if [ -f "$src" ]; then
        ln -sf "$src" "$dst"
        echo "  Linked: $dst -> $src"
    fi
done

# Create config template if not exists
CONF_TEMPLATE="$MONITOR_DIR/taskctl.conf"
if [ ! -f "$CONF_TEMPLATE" ]; then
    cat > "$CONF_TEMPLATE" << 'EOF'
# taskctl.conf — Local configuration overrides
# This file is NOT committed to git.
#
# QQ_TARGET_USER=your_qq_user_id
EOF
    echo "  Created: $CONF_TEMPLATE (edit me!)"
fi

# Crontab setup (interactive, ask user)
echo ""
echo "[?] Add crontab entries for automatic monitoring?"
echo "  */15 * * * * cd $MONITOR_DIR && python3 taskctl.py check >/dev/null 2>&1"
echo "  0 4 * * * cd $MONITOR_DIR && python3 taskctl.py clear >/dev/null 2>&1"
echo ""
echo "Add them manually with: crontab -e"
echo ""

echo "[OK] Installation complete."
echo ""
echo "Quick start:"
echo "  # Run a command and register it"
echo "  long_running_script.py &"
echo "  PID=\$!"
echo "  python3 $MONITOR_DIR/taskctl.py add \\"
echo "    --id my_task --pid \$PID --ctx \"description\" --duration 3600 \\"
echo "    --on-success \"expflow analyze advise --task task1\""
echo ""
echo "  # Manually check"
echo "  python3 $MONITOR_DIR/taskctl.py list"
echo "  python3 $MONITOR_DIR/taskctl.py status"
