#!/bin/bash
# Double-click in Finder to play a recorded session (see play_session.py)
cd "$(dirname "$0")" && .venv/bin/python play_session.py "$@"
echo
read -n 1 -s -r -p "Press any key to close this window"
