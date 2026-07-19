#!/usr/bin/env bash
set -euo pipefail

api_base="${API_BASE_URL:-http://localhost:8000}"
web_base="${WEB_BASE_URL:-http://localhost:3000}"
generated_collection="smoke_mask_$(date -u +%Y%m%d%H%M%S)_${RANDOM}_$$"
smoke_collection="${SMOKE_COLLECTION_PREFIX:-$generated_collection}"
run_ma_yun_flow="${RUN_MA_YUN_FLOW:-0}"
skip_build="${SMOKE_SKIP_BUILD:-0}"

if (( ${#smoke_collection} < 3 || ${#smoke_collection} > 512 )) \
  || [[ ! "$smoke_collection" =~ ^[A-Za-z0-9] ]] \
  || [[ ! "$smoke_collection" =~ [A-Za-z0-9]$ ]] \
  || [[ "$smoke_collection" =~ [^A-Za-z0-9._-] ]] \
  || [[ "$smoke_collection" == *".."* ]]; then
  echo "SMOKE_COLLECTION_PREFIX is not a valid Chroma collection name." >&2
  exit 2
fi

if [[ "$run_ma_yun_flow" != "0" && "$run_ma_yun_flow" != "1" ]]; then
  echo "RUN_MA_YUN_FLOW must be 0 or 1." >&2
  exit 2
fi

if [[ "$skip_build" != "0" && "$skip_build" != "1" ]]; then
  echo "SMOKE_SKIP_BUILD must be 0 or 1." >&2
  exit 2
fi

if ! docker compose config --quiet >/dev/null 2>&1; then
  echo "Docker Compose configuration is incomplete. Add OPENAI_API_KEY to the ignored .env file first." >&2
  exit 1
fi

on_error() {
  status=$?
  echo "Smoke test failed (exit $status). Recent backend logs follow; secrets are never printed by this script." >&2
  docker compose ps >&2 || true
  docker compose logs --tail=120 backend >&2 || true
  exit "$status"
}
trap on_error ERR

# A unique collection prevents old prompt, signature, or fixture rows from turning
# the intended fresh first request into a cache hit.  The variable is expanded by
# Compose and is injected only into the backend container.
compose_up=(up -d --force-recreate backend frontend)
if [[ "$skip_build" == "0" ]]; then
  compose_up=(up --build -d --force-recreate backend frontend)
fi
CHROMA_COLLECTION_PREFIX="$smoke_collection" QUERY_SIGNATURE_VERSION=5 \
  docker compose "${compose_up[@]}"

ready=0
for _attempt in $(seq 1 120); do
  if curl -fsS "$api_base/ready" >/dev/null 2>&1 \
    && curl -fsS "$web_base" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "Services did not become ready within 120 seconds." >&2
  exit 1
fi

curl -fsS "$api_base/health" >/dev/null
curl -fsS "$api_base/ready" >/dev/null
curl -fsS "$web_base" | grep -q '<div id="root"></div>'

# Verify that the image contains exactly the three raw JSON sources and that the
# source arrays retain their required row counts.  No generated facts are accepted.
docker compose exec -T backend python -c '
import json
from pathlib import Path

root = Path("/app/data")
actual = {item.name for item in root.iterdir() if item.suffix == ".json"}
expected = {"person 1.json", "company 1.json", "relations 1.json"}
assert actual == expected, (actual, expected)
assert len(json.loads((root / "person 1.json").read_text())) == 20
assert len(json.loads((root / "company 1.json").read_text())) == 30
assert len(json.loads((root / "relations 1.json").read_text())) == 109
'

first_response=$(curl -fsS -X POST "$api_base/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Mask控制了哪些公司？","locale":"zh-CN"}')

conversation_id=$(printf '%s' "$first_response" | python3 -c '
import json, sys

body = json.load(sys.stdin)
assert body["status"] == "success" and body["error_code"] is None
labels = {node["label"] for node in body["graph"]["nodes"]}
assert {"Elon Musk", "Tesla, Inc.", "SpaceX", "xAI"} <= labels, labels

expected = {
    ("person:P001", "company:C001", "works_at", "CEO_of"),
    ("person:P001", "company:C002", "founded", "Founder_of"),
    ("person:P001", "company:C021", "founded", "Founder_of"),
}
actual = {
    (
        edge["source"],
        edge["target"],
        edge["type"],
        edge["properties"].get("raw_relation"),
    )
    for edge in body["graph"]["edges"]
    if edge["source"] == "person:P001"
    and edge["target"] in {"company:C001", "company:C002", "company:C021"}
}
assert actual == expected, actual
assert not [edge for edge in body["graph"]["edges"] if edge["type"] == "controls"]

answer = body["answer"]
assert "显式" in answer and "控制" in answer, answer
assert "不等同" in answer and "法律控制" in answer, answer

memory = body["memory"]
assert memory["cache_hit"] is False, memory
assert memory["status"] == "warm", memory
assert memory["write_operation"] == "add", memory

trace = body["trace"]
assert trace["model_provider"] == "openai", trace
assert trace["model_calls"] > 0, trace
assert trace["planner_model_calls"] > 0, trace
assert trace["researcher_model_calls"] > 0, trace
assert trace["visualizer_model_calls"] > 0, trace
assert trace["researcher_invoked"] is True, trace
assert trace["tool_calls"] > 0, trace
roles = {step["role"] for step in trace["agent_steps"]}
assert {"planner", "researcher", "visualizer"} <= roles, roles
tools = {step.get("tool") for step in trace["agent_steps"] if step.get("tool")}
assert {"persons", "relations"} <= tools, tools
assert "planner" in trace["prompt_versions"], trace
assert "researcher" in trace["prompt_versions"], trace
assert "visualizer" in trace["prompt_versions"], trace
print(body["conversation_id"])
')

second_response=$(curl -fsS -X POST "$api_base/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"conversation_id\":\"$conversation_id\",\"message\":\"这些公司在哪？\",\"locale\":\"zh-CN\"}")

printf '%s' "$second_response" | python3 -c '
import json, sys

body = json.load(sys.stdin)
assert body["status"] == "success" and body["error_code"] is None
locations = {
    node["label"]
    for node in body["graph"]["nodes"]
    if node["type"] == "location"
}
assert {"Austin", "Hawthorne", "San Francisco"} <= locations, locations
location_edges = [
    edge
    for edge in body["graph"]["edges"]
    if edge["type"] == "headquartered_in"
]
expected_sources = {"company:C001", "company:C002", "company:C021"}
assert expected_sources <= {edge["source"] for edge in location_edges}, location_edges
trace = body["trace"]
assert trace["researcher_invoked"] is True, trace
assert trace["tool_calls"] > 0, trace
assert {"planner", "researcher", "visualizer"} <= {
    step["role"] for step in trace["agent_steps"]
}, trace["agent_steps"]
'

third_response=$(curl -fsS -X POST "$api_base/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"conversation_id\":\"$conversation_id\",\"message\":\"Mask控制了哪些公司？\",\"locale\":\"zh-CN\"}")

graph_id=$(printf '%s' "$third_response" | python3 -c '
import json, sys

body = json.load(sys.stdin)
assert body["status"] == "success" and body["error_code"] is None
memory = body["memory"]
assert memory["cache_hit"] is True, memory
assert memory["match_type"] == "raw_exact", memory
assert memory["status"] == "hot", memory
trace = body["trace"]
assert trace["researcher_invoked"] is False, trace
assert trace["tool_calls"] == 0, trace
assert trace["model_calls"] == 0, trace
assert trace["planner_model_calls"] == 0, trace
assert trace["researcher_model_calls"] == 0, trace
assert trace["visualizer_model_calls"] == 0, trace
assert trace["agent_steps"] == [], trace
print(body["graph_id"])
')

curl -fsS "$api_base/graph?conversation_id=$conversation_id" >/dev/null
curl -fsS "$api_base/graph?graph_id=$graph_id" >/dev/null

run_ma_yun_flow() {
  local first second third ma_yun_conversation

  first=$(curl -fsS -X POST "$api_base/chat" \
    -H 'Content-Type: application/json' \
    -d '{"message":"马云创办了哪些公司？","locale":"zh-CN"}')

  ma_yun_conversation=$(printf '%s' "$first" | python3 -c '
import json, sys

body = json.load(sys.stdin)
assert body["status"] == "success" and body["error_code"] is None
labels = {node["label"] for node in body["graph"]["nodes"]}
assert {"马云", "阿里巴巴集团"} <= labels, labels
founded = [
    edge for edge in body["graph"]["edges"]
    if edge["source"] == "person:P004"
    and edge["target"] == "company:C005"
    and edge["type"] == "founded"
]
assert founded, founded
assert all(edge["properties"]["raw_relation"] == "Founder_of" for edge in founded), founded
assert body["memory"]["cache_hit"] is False, body["memory"]
assert body["memory"]["status"] == "warm", body["memory"]
assert body["trace"]["model_calls"] > 0, body["trace"]
assert body["trace"]["tool_calls"] > 0, body["trace"]
print(body["conversation_id"])
')

  second=$(curl -fsS -X POST "$api_base/chat" \
    -H 'Content-Type: application/json' \
    -d "{\"conversation_id\":\"$ma_yun_conversation\",\"message\":\"这些公司在哪？\",\"locale\":\"zh-CN\"}")

  printf '%s' "$second" | python3 -c '
import json, sys

body = json.load(sys.stdin)
assert body["status"] == "success" and body["error_code"] is None
locations = {
    node["label"]
    for node in body["graph"]["nodes"]
    if node["type"] == "location"
}
assert "Hangzhou" in locations, locations
'

  third=$(curl -fsS -X POST "$api_base/chat" \
    -H 'Content-Type: application/json' \
    -d "{\"conversation_id\":\"$ma_yun_conversation\",\"message\":\"马云创办了哪些公司？\",\"locale\":\"zh-CN\"}")

  printf '%s' "$third" | python3 -c '
import json, sys

body = json.load(sys.stdin)
assert body["status"] == "success" and body["error_code"] is None
assert body["memory"]["cache_hit"] is True, body["memory"]
assert body["memory"]["match_type"] == "raw_exact", body["memory"]
assert body["memory"]["status"] == "hot", body["memory"]
trace = body["trace"]
assert trace["model_calls"] == 0, trace
assert trace["tool_calls"] == 0, trace
assert trace["researcher_invoked"] is False, trace
'
}

if [[ "$run_ma_yun_flow" == "1" ]]; then
  run_ma_yun_flow
fi

docker compose logs backend | grep -q 'cache_hit'

trap - ERR
printf 'Smoke test passed: conversation=%s graph=%s collection=%s ma_yun_flow=%s\n' \
  "$conversation_id" "$graph_id" "$smoke_collection" "$run_ma_yun_flow"
