#!/usr/bin/env bash
# First-time DGX Spark setup for spark-lab, as one guided run. Run it once over SSH, as your
# normal user, from the checkout at ~/lab/spark-lab:
#
#   bash scripts/bootstrap.sh
#
# It is safe to rerun: every step checks first and skips what is already done.
# It never runs sudo. Steps that need root are printed for you to run in another SSH session;
# the script waits, re-checks, and moves on (or you type "skip").
#
# Steps: machine checks -> sudo steps -> ~/.config/spark-lab.env -> docker images -> teacher
# weights -> user services -> wait for /v1/models -> Roman Urdu smoke test -> smoke job -> checklist.
# Sources for every image tag and setting are in docs/bootstrap.md.
set -euo pipefail

# Verified 2026-11-03 against NVIDIA's DGX Spark vLLM playbook (Gemma 4 family uses this image).
# The digest pins the multi-arch index; the arm64 image inside it is what the Spark pulls.
VERIFIED_VLLM_IMAGE="vllm/vllm-openai:gemma4-cu130@sha256:0d152595cd940ea1e0890fa190a37b7a86e3f7f0f5048ac3e2e4c14529fea833"
VERIFIED_TEACHER_MODEL="google/gemma-4-31B-it"
# Base image of NVIDIA's DGX Spark Unsloth playbook; training jobs install Unsloth on top of it.
UNSLOTH_BASE_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"

LAB="${LAB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${SPARK_LAB_ENV:-$HOME/.config/spark-lab.env}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
HF_CACHE="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
TEACHER_URL_OVERRIDE="${TEACHER_URL:-}"       # set in the environment: wins over the env file
TEACHER_URL="${TEACHER_URL:-http://localhost:8000/v1}"
TEACHER_WAIT_MIN="${TEACHER_WAIT_MIN:-45}"   # first start includes loading ~62 GB of weights
POLL_SEC="${BOOTSTRAP_POLL_SEC:-15}"
JOB_WAIT_SEC="${BOOTSTRAP_JOB_WAIT_SEC:-120}"
MODEL_MIN_FREE_GB="${MODEL_MIN_FREE_GB:-100}"
SSHD_CONFIG_DIR="${SSHD_CONFIG_DIR:-/etc/ssh}"
PY="${LAB_PYTHON:-python3}"
NOTIFY="${BOOTSTRAP_NOTIFY:-1}"                # 0 skips the final phone alert (tests)

DONE_ITEMS=()
TODO_ITEMS=()
FAILED=0
ENV_CHANGED=0

if [ -t 1 ]; then C_OK=$'\033[32m' C_TODO=$'\033[33m' C_ERR=$'\033[31m' C_B=$'\033[1m' C_0=$'\033[0m'
else C_OK='' C_TODO='' C_ERR='' C_B='' C_0=''; fi

step() { printf '\n%s== %s%s\n' "$C_B" "$*" "$C_0"; }
ok()   { printf '  %sok%s    %s\n' "$C_OK" "$C_0" "$*"; DONE_ITEMS+=("$*"); }
info() { printf '        %s\n' "$*"; }
todo() { printf '  %stodo%s  %s\n' "$C_TODO" "$C_0" "$*"; TODO_ITEMS+=("$*"); }
fail() { printf '  %sFAIL%s  %s\n' "$C_ERR" "$C_0" "$*"; TODO_ITEMS+=("FAILED: $*"); FAILED=1; }
die()  { printf '%serror:%s %s\n' "$C_ERR" "$C_0" "$*" >&2; exit 1; }

# ask PROMPT -> sets REPLY_TEXT; returns 1 on end of input (non-interactive run).
ask() {
  printf '%s' "$1" >&2
  IFS= read -r REPLY_TEXT || { printf '\n' >&2; return 1; }
}

# ask_secret PROMPT -> like ask, without echoing what is typed.
ask_secret() {
  printf '%s' "$1" >&2
  if [ -t 0 ]; then
    IFS= read -rs REPLY_TEXT || { printf '\n' >&2; return 1; }
    printf '\n' >&2
  else
    IFS= read -r REPLY_TEXT || return 1
  fi
}

# ---------------------------------------------------------------------------------------------
# sudo steps: print the exact commands, wait for the owner to run them, re-check.

# sudo_step DESCRIPTION CHECK_FUNCTION COMMAND...
sudo_step() {
  local what="$1" check="$2"
  shift 2
  if "$check"; then ok "$what"; return 0; fi
  printf '  %ssudo%s  %s: run this yourself in another SSH session:\n' "$C_TODO" "$C_0" "$what"
  local line
  for line in "$@"; do printf '          %s\n' "$line"; done
  while :; do
    if ! ask "        Press Enter when done to re-check, or type 'skip': "; then REPLY_TEXT=skip; fi
    if [ "$REPLY_TEXT" = skip ]; then
      todo "$what (skipped; run: $(printf '%s; ' "$@" | sed 's/; $//'))"
      return 0
    fi
    if "$check"; then ok "$what"; return 0; fi
    info "not done yet"
  done
}

check_docker_group()  { docker info >/dev/null 2>&1 || id -nG "$USER" | tr ' ' '\n' | grep -qx docker; }
check_linger()        { [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = yes ]; }
check_no_sleep()      { systemctl is-enabled sleep.target 2>/dev/null | grep -q masked; }
check_tmux()          { command -v tmux >/dev/null; }
check_agent_user()    { getent passwd agent >/dev/null; }
check_ssh_keys_only() {
  grep -Eqs '^\s*PasswordAuthentication\s+no' "$SSHD_CONFIG_DIR"/sshd_config "$SSHD_CONFIG_DIR"/sshd_config.d/*.conf
}

# The user's systemd manager keeps the groups it started with (and linger keeps it alive),
# so after joining the docker group it needs a restart before user services can use docker.
check_user_manager_docker() {
  local gid pid
  gid="$(getent group docker | cut -d: -f3)" || return 0
  [ -n "$gid" ] || return 0
  pid="$(systemctl show "user@$(id -u).service" -p MainPID --value 2>/dev/null)" || return 0
  [ -n "$pid" ] && [ "$pid" != 0 ] && [ -r "/proc/$pid/status" ] || return 0
  grep '^Groups:' "/proc/$pid/status" | tr -s ' \t' '\n' | grep -qx "$gid"
}

sudo_steps() {
  step "Steps that need sudo (this script never runs sudo itself)"
  sudo_step "docker without sudo" check_docker_group \
    "sudo usermod -aG docker $USER"
  if ! docker info >/dev/null 2>&1; then
    todo "docker still not usable: if you just joined the docker group, log out, SSH in again and rerun bash scripts/bootstrap.sh (otherwise: systemctl status docker)"
    checklist
    exit 3
  fi
  sudo_step "user services keep running after logout (linger)" check_linger \
    "sudo loginctl enable-linger $USER"
  sudo_step "user service manager is in the docker group" check_user_manager_docker \
    "sudo systemctl restart user@$(id -u).service"
  sudo_step "Spark never sleeps" check_no_sleep \
    "sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target"
  sudo_step "tmux installed" check_tmux \
    "sudo apt-get install -y tmux"
  sudo_step "agent user (no sudo group) for coding agents" check_agent_user \
    "sudo adduser --disabled-password --gecos '' agent"
  info "Check that SSH key login works in a second session before turning passwords off."
  sudo_step "SSH password login off" check_ssh_keys_only \
    "echo 'PasswordAuthentication no' | sudo tee $SSHD_CONFIG_DIR/sshd_config.d/10-keys-only.conf" \
    "sudo systemctl reload ssh"
}

# ---------------------------------------------------------------------------------------------
# Machine checks

preflight() {
  step "Machine"
  [ "$(id -u)" != 0 ] || die "run this as your normal user, not as root or with sudo"
  local c
  for c in docker curl "$PY" systemctl; do
    command -v "$c" >/dev/null || die "$c not found; DGX OS ships it, so check PATH"
  done
  local arch; arch="$(uname -m)"
  if [ "$arch" = aarch64 ]; then ok "ARM64 ($arch)"; else todo "expected aarch64 (DGX Spark), got $arch"; fi
  if command -v nvidia-smi >/dev/null; then
    local gpu driver
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    ok "GPU ${gpu:-?}, driver ${driver:-?}"
    # The gemma4-cu130 image is built on CUDA 13.0, which needs driver 580 or newer.
    if [ -n "$driver" ] && [ "${driver%%.*}" -lt 580 ] 2>/dev/null; then
      todo "driver $driver is older than 580 (CUDA 13.0): update DGX OS through the DGX Dashboard"
    fi
  else
    fail "nvidia-smi missing: is this the Spark?"
  fi
  if [ "$LAB" = "$HOME/lab/spark-lab" ]; then ok "checkout at ~/lab/spark-lab"
  else todo "the services expect the checkout at ~/lab/spark-lab, it is at $LAB (move it and rerun)"; fi
}

# ---------------------------------------------------------------------------------------------
# ~/.config/spark-lab.env

env_get() {  # env_get KEY -> value of the last KEY= line, surrounding quotes removed
  [ -f "$ENV_FILE" ] || return 0
  sed -n "s/^$1=//p" "$ENV_FILE" | tail -1 | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/"
}

env_set() {  # env_set KEY VALUE (value must not contain newlines or backslashes)
  local key="$1" val="$2" tmp
  [ "$(env_get "$key")" = "$val" ] && grep -q "^$key=" "$ENV_FILE" && return 0
  tmp="$(mktemp "$ENV_FILE.XXXXXX")"
  awk -v k="$key" -v v="$val" '
    index($0, k "=") == 1 { if (!done) print k "=" v; done = 1; next }
    { print }
    END { if (!done) print k "=" v }' "$ENV_FILE" >"$tmp"
  chmod 600 "$tmp"
  mv "$tmp" "$ENV_FILE"
  ENV_CHANGED=1
}

valid_topic() { [[ "$1" =~ ^[A-Za-z0-9_-]{8,64}$ ]] && [[ "$1" != *CHANGE-ME* ]]; }
valid_hf_token() { [[ "$1" =~ ^hf_[A-Za-z0-9]{20,}$ ]]; }
random_topic() { printf 'spark-lab-%s' "$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')"; }

hf_whoami() {  # hf_whoami TOKEN -> prints the account name; the token goes through stdin, not argv
  printf 'Authorization: Bearer %s\n' "$1" \
    | curl -sSf -m 15 -H @- "$HF_ENDPOINT/api/whoami-v2" 2>/dev/null \
    | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("name", "?"))' 2>/dev/null
}

write_env() {
  step "Config: $ENV_FILE"
  mkdir -p "$(dirname "$ENV_FILE")"
  if [ ! -f "$ENV_FILE" ]; then
    (umask 077 && cp "$LAB/systemd/spark-lab.env.example" "$ENV_FILE")
    ENV_CHANGED=1
    ok "created from systemd/spark-lab.env.example"
  fi
  chmod 600 "$ENV_FILE"

  local topic; topic="$(env_get NTFY_TOPIC)"
  if valid_topic "$topic"; then
    ok "NTFY_TOPIC already set"
  else
    local suggestion; suggestion="$(random_topic)"
    info "Phone alerts use ntfy: install the ntfy app and subscribe to this topic."
    info "Anyone who knows the topic can read the alerts, so keep it long and random."
    while :; do
      ask "        NTFY topic [$suggestion]: " || REPLY_TEXT=""
      topic="${REPLY_TEXT:-$suggestion}"
      valid_topic "$topic" && break
      info "8-64 characters from A-Z a-z 0-9 _ -"
    done
    env_set NTFY_TOPIC "$topic"
    ok "NTFY_TOPIC=$topic (subscribe to it in the ntfy app)"
  fi

  local token; token="$(env_get HF_TOKEN)"
  if [ -n "$token" ]; then
    ok "HF_TOKEN already set"
  else
    info "Hugging Face READ token (huggingface.co/settings/tokens, type 'Read')."
    info "Gemma 4 is not gated, so this only avoids rate limits; leave blank to skip."
    while :; do
      ask_secret "        HF read token (hidden): " || REPLY_TEXT=""
      token="$REPLY_TEXT"
      [ -z "$token" ] && break
      valid_hf_token "$token" && break
      info "that does not look like a Hugging Face token (hf_...)"
    done
    if [ -n "$token" ]; then
      env_set HF_TOKEN "$token"
      ok "HF_TOKEN saved"
    else
      todo "no HF_TOKEN: add a read token to $ENV_FILE if downloads get rate-limited"
    fi
  fi
  if [ -n "$token" ]; then
    local who; who="$(hf_whoami "$token" || true)"
    if [ -n "$who" ]; then ok "HF token works (account: $who)"
    else todo "could not verify HF_TOKEN against $HF_ENDPOINT (offline or revoked?)"; fi
  fi

  local image; image="$(env_get VLLM_IMAGE)"
  if [ -z "$image" ] || [[ "$image" == *CHECK-PLAYBOOK* ]]; then
    env_set VLLM_IMAGE "$VERIFIED_VLLM_IMAGE"
    ok "VLLM_IMAGE=$VERIFIED_VLLM_IMAGE"
  elif [ "$image" = "$VERIFIED_VLLM_IMAGE" ]; then
    ok "VLLM_IMAGE is the verified image"
  else
    ok "VLLM_IMAGE=$image (kept your choice; verified: $VERIFIED_VLLM_IMAGE)"
  fi

  local model; model="$(env_get TEACHER_MODEL)"
  # The old example had the wrong case; vLLM serves the id exactly as given.
  if [ -z "$model" ] || [ "$model" = google/gemma-4-31b-it ]; then
    env_set TEACHER_MODEL "$VERIFIED_TEACHER_MODEL"
    model="$VERIFIED_TEACHER_MODEL"
  fi
  ok "TEACHER_MODEL=$model"

  [ -n "$(env_get VLLM_GPU_UTIL)" ] || env_set VLLM_GPU_UTIL 0.70
  [ -n "$(env_get TEACHER_URL)" ] || env_set TEACHER_URL "$TEACHER_URL"
  ok "VLLM_GPU_UTIL=$(env_get VLLM_GPU_UTIL), TEACHER_URL=$(env_get TEACHER_URL)"

  # Use the file's values for the rest of this run.
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +a
  TEACHER_URL="${TEACHER_URL_OVERRIDE:-$(env_get TEACHER_URL)}"
}

# ---------------------------------------------------------------------------------------------
# Images and weights

pull_image() {
  local image="$1" why="$2"
  if docker image inspect "$image" >/dev/null 2>&1; then
    ok "$why image present: $image"
  elif docker pull "$image"; then
    ok "$why image pulled: $image"
  else
    fail "could not pull $image"
  fi
}

images() {
  step "Container images"
  pull_image "$VLLM_IMAGE" "teacher (vLLM)"
  pull_image "$UNSLOTH_BASE_IMAGE" "training base (Unsloth playbook)"
}

free_gb() { df -Pk "$1" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'; }

download_weights() {
  step "Teacher weights: $TEACHER_MODEL"
  mkdir -p "$HF_CACHE"
  local free; free="$(free_gb "$HF_CACHE")"
  if [ "$free" -lt "$MODEL_MIN_FREE_GB" ]; then
    todo "only ${free} GB free under $HF_CACHE; the BF16 weights need about 63 GB"
  fi
  info "Downloading into $HF_CACHE (resumes and skips files already there)."
  # Same image and cache directory as vllm-teacher.service, so the service finds the files.
  # Runs as you, not root, so the cache stays writable for your own training jobs.
  if docker run --rm --entrypoint python3 \
      --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
      -e HF_TOKEN -e TEACHER_MODEL \
      -v "$HF_CACHE:/hf" \
      "$VLLM_IMAGE" \
      -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["TEACHER_MODEL"]))'; then
    ok "weights cached"
  else
    fail "weight download failed (rerun to resume)"
  fi
}

# ---------------------------------------------------------------------------------------------
# User services

install_units() {
  step "User services ($UNIT_DIR)"
  mkdir -p "$UNIT_DIR"
  local f name changed=0
  for f in "$LAB"/systemd/*.service "$LAB"/systemd/*.timer; do
    name="$(basename "$f")"
    if ! cmp -s "$f" "$UNIT_DIR/$name"; then
      cp "$f" "$UNIT_DIR/$name"
      changed=1
      info "installed $name"
    fi
  done
  if [ $changed = 1 ]; then systemctl --user daemon-reload; fi
  local u was_active
  for u in lab-nightly.timer lab-worker.service vllm-teacher.service; do
    was_active=0
    systemctl --user is-active --quiet "$u" && was_active=1
    if ! systemctl --user enable --now "$u" >/dev/null 2>&1; then
      fail "$u would not start: systemctl --user status $u"
      continue
    fi
    # A service that was already running keeps its old unit/env until restarted. A running job
    # is put back in the queue and resumes from its last checkpoint.
    if [ $was_active = 1 ] && [[ "$u" == *.service ]] && { [ $changed = 1 ] || [ $ENV_CHANGED = 1 ]; }; then
      systemctl --user restart "$u"
      ok "$u restarted to pick up changes"
    else
      ok "$u enabled"
    fi
  done
}

teacher_lists_model() {
  curl -sf -m 5 --noproxy '*' "$TEACHER_URL/models" 2>/dev/null \
    | M="$TEACHER_MODEL" "$PY" -c 'import json,os,sys; ids=[m["id"] for m in json.load(sys.stdin)["data"]]; sys.exit(os.environ["M"] not in ids)' 2>/dev/null
}

wait_for_teacher() {
  step "Waiting for the teacher at $TEACHER_URL/models (up to $TEACHER_WAIT_MIN min)"
  local deadline=$((SECONDS + TEACHER_WAIT_MIN * 60)) last=-1
  until teacher_lists_model; do
    if systemctl --user is-failed --quiet vllm-teacher.service; then
      fail "vllm-teacher.service failed; last log lines:"
      docker logs --tail 20 vllm-teacher 2>&1 | sed 's/^/          /' || true
      info "full log: journalctl --user -u vllm-teacher -n 200"
      return 1
    fi
    if [ $SECONDS -ge $deadline ]; then
      fail "teacher not ready after $TEACHER_WAIT_MIN min: journalctl --user -u vllm-teacher -n 200"
      return 1
    fi
    if [ $((SECONDS / 60)) != "$last" ]; then
      last=$((SECONDS / 60))
      info "$(date +%H:%M) $(docker logs --tail 1 vllm-teacher 2>&1 | cut -c1-110 || true)"
    fi
    sleep "$POLL_SEC"
  done
  ok "teacher serving $TEACHER_MODEL"
}

teacher_smoke() {
  step "Roman Urdu smoke test (3 prompts)"
  if (cd "$LAB" && "$PY" -m lab teacher-smoke --base-url "$TEACHER_URL" --model "$TEACHER_MODEL"); then
    ok "teacher answers in Roman Urdu"
  else
    fail "teacher smoke test failed (output above)"
  fi
}

# ---------------------------------------------------------------------------------------------
# Queue

smoke_job_seen() {
  local root="${LAB_ROOT:-$LAB}" d
  for d in queue running "done"; do
    compgen -G "$root/jobs/$d/*-smoke-python.json" >/dev/null && { printf '%s' "$d"; return 0; }
  done
  return 1
}

enqueue_smoke() {
  step "Smoke job"
  local where
  if where="$(smoke_job_seen)"; then
    ok "smoke job already in jobs/$where/"
  elif (cd "$LAB" && "$PY" -m lab enqueue jobs/nightly.d/example-smoke.json >/dev/null); then
    ok "smoke job queued"
  else
    fail "could not enqueue jobs/nightly.d/example-smoke.json"
    return 0
  fi
  if systemctl --user is-active --quiet lab-worker.service; then
    local deadline=$((SECONDS + JOB_WAIT_SEC))
    while [ $SECONDS -lt $deadline ] && [ "$(smoke_job_seen)" != "done" ]; do sleep 5; done
    if [ "$(smoke_job_seen)" = "done" ]; then ok "worker ran the smoke job"
    else todo "smoke job not finished yet: python3 -m lab status"; fi
  else
    todo "lab-worker.service is not running, so the smoke job waits: systemctl --user status lab-worker"
  fi
}

# ---------------------------------------------------------------------------------------------

checklist() {
  step "Checklist"
  local i
  for i in "${DONE_ITEMS[@]}"; do printf '  [x] %s\n' "$i"; done
  for i in "${TODO_ITEMS[@]}"; do printf '  [ ] %s\n' "$i"; done
  cat <<EOF

  By hand:
  [ ] Phone got the "spark-lab bootstrap" alert (ntfy topic $(env_get NTFY_TOPIC))
  [ ] Spark and router are on a UPS
  [ ] Next nightly run is listed: systemctl --user list-timers lab-nightly.timer
  [ ] Queue looks right: python3 -m lab status
  [ ] Before big training runs: systemctl --user stop vllm-teacher (frees ~90 GB)

EOF
  if [ $FAILED = 1 ]; then echo "Some steps failed. Fix them and rerun: bash scripts/bootstrap.sh"
  elif [ ${#TODO_ITEMS[@]} -gt 0 ]; then echo "Set up, with the open items above. Rerun any time; done steps are skipped."
  else echo "All set."; fi
}

usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

main() {
  case "${1:-}" in
    -h|--help) usage; return 0 ;;
    "") ;;
    *) usage >&2; return 2 ;;
  esac
  cd "$LAB"
  preflight
  sudo_steps
  write_env
  images
  download_weights
  install_units
  if wait_for_teacher; then teacher_smoke; fi
  enqueue_smoke
  [ "$NOTIFY" = 0 ] || (cd "$LAB" && "$PY" -m lab notify "spark-lab bootstrap finished on $(hostname)" >/dev/null 2>&1) || true
  checklist
  [ $FAILED = 0 ]
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
