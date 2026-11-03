#!/usr/bin/env bash
# One-time Spark setup. Run on the Spark as your normal user from the spark-lab checkout:
#   bash scripts/setup_spark.sh            # checks only, changes nothing
#   bash scripts/setup_spark.sh --apply    # also installs user services (no sudo)
# Steps that need sudo are printed for you to run yourself; this script never calls sudo.
# For the full guided first-time setup (env file, images, teacher, smoke tests) use
# scripts/bootstrap.sh instead; this script stays as a quick read-only health check.
# ok/todo always succeed, so `check && ok || todo` is a safe if/else here.
# shellcheck disable=SC2015
set -uo pipefail

APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
todo() { printf '  \033[33mtodo\033[0m  %s\n' "$*"; }
LAB="$(cd "$(dirname "$0")/.." && pwd)"

echo "== Machine"
[ "$(uname -m)" = "aarch64" ] && ok "ARM64 ($(uname -m))" || todo "expected aarch64, got $(uname -m)"
command -v nvidia-smi >/dev/null && ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" || todo "nvidia-smi missing"
command -v docker >/dev/null && ok "docker" || todo "install docker (DGX OS ships it)"
command -v tmux >/dev/null && ok "tmux" || todo "sudo apt install -y tmux"
command -v tailscale >/dev/null && ok "tailscale $(tailscale ip -4 2>/dev/null | head -1)" || todo "install Tailscale"

echo "== Stay awake (needs sudo; run these yourself)"
if systemctl is-enabled sleep.target 2>/dev/null | grep -q masked; then ok "sleep targets masked"
else todo "sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target"; fi
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; then ok "linger on (user services run without login)"
else todo "sudo loginctl enable-linger $USER"; fi

echo "== SSH"
if grep -Eqs '^\s*PasswordAuthentication\s+no' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf; then ok "password login off"
else todo "set 'PasswordAuthentication no' in /etc/ssh/sshd_config.d/10-keys-only.conf, then: sudo systemctl reload ssh"; fi

echo "== Agent user"
if id agent >/dev/null 2>&1; then ok "user 'agent' exists"
else todo "sudo adduser --disabled-password --gecos '' agent   # no sudo group; agents run as this user"; fi

echo "== Config"
ENV="$HOME/.config/spark-lab.env"
if [ -f "$ENV" ]; then ok "$ENV"
else
  todo "cp systemd/spark-lab.env.example $ENV && chmod 600 $ENV, then edit it"
  [ $APPLY = 1 ] && mkdir -p "$HOME/.config" && cp "$LAB/systemd/spark-lab.env.example" "$ENV" && chmod 600 "$ENV" && ok "copied example env (or run scripts/bootstrap.sh to fill it in)"
fi
command -v hf >/dev/null && ok "hf CLI" || todo "pip install -U huggingface_hub   # then: hf auth login (read token)"

echo "== Services"
UNIT_DIR="$HOME/.config/systemd/user"
if [ $APPLY = 1 ]; then
  mkdir -p "$UNIT_DIR"
  cp "$LAB"/systemd/*.service "$LAB"/systemd/*.timer "$UNIT_DIR"/
  systemctl --user daemon-reload
  systemctl --user enable --now lab-nightly.timer && ok "lab-nightly.timer enabled (23:00)"
  systemctl --user enable --now lab-worker.service && ok "lab-worker.service running"
  todo "bash scripts/bootstrap.sh downloads the weights and starts vllm-teacher.service"
else
  for u in lab-worker.service lab-nightly.timer vllm-teacher.service; do
    systemctl --user is-enabled "$u" >/dev/null 2>&1 && ok "$u enabled" || todo "$u not installed (rerun with --apply)"
  done
fi

echo "== Teacher"
if curl -sf -m 3 http://localhost:8000/v1/models >/dev/null; then ok "teacher answering on :8000"
else todo "teacher not answering on localhost:8000 yet"; fi

echo
echo "Also: put the Spark and router on a UPS, then test: python3 -m lab notify 'hello'"
