#!/usr/bin/env bash
set -euo pipefail
ARTIFACT_DIR=/tmp/sfn-release-artifacts
CANDIDATE_IMAGE=scalarforensic:release-check-20260908
docker run --rm --network none -v "$ARTIFACT_DIR/runtime-results:/results" alpine:latest chown 10001:10001 /results
docker run --rm --gpus all --network none -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -v "$ARTIFACT_DIR/evidence:/images:ro" -v "$ARTIFACT_DIR/model_smoke.py:/model_smoke.py:ro" "$CANDIDATE_IMAGE" python /model_smoke.py > "$ARTIFACT_DIR/model-smoke-final.log" 2>&1
docker run --name sfn-release-ingest --gpus all --network sfn-release-offline-20260908 --env-file "$ARTIFACT_DIR/runtime.env" -v "$ARTIFACT_DIR/evidence:/images:ro" -v "$ARTIFACT_DIR/source-state:/app/data" "$CANDIDATE_IMAGE" sfn /images --dino --sscd > "$ARTIFACT_DIR/ingest.log" 2>&1
docker run -d --name sfn-release-web-source --network sfn-release-offline-20260908 --network-alias release-web --env-file "$ARTIFACT_DIR/runtime.env" -e SFN_DEVICE=cpu -v "$ARTIFACT_DIR/evidence:/images:ro" -v "$ARTIFACT_DIR/source-state:/app/data" "$CANDIDATE_IMAGE"
for i in $(seq 1 30); do
    if docker exec sfn-release-web-source python -c 'import urllib.request; assert urllib.request.urlopen("http://localhost:8080/").status == 200' 2>/dev/null; then break; fi
    if [ "$i" = 30 ]; then exit 1; fi
    sleep 2
done
docker run --rm --network sfn-release-offline-20260908 --env-file "$ARTIFACT_DIR/runtime.env" -v "$ARTIFACT_DIR/evidence:/images:ro" -v "$ARTIFACT_DIR/source-state:/app/data:ro" -v "$ARTIFACT_DIR/verify_runtime.py:/verify_runtime.py:ro" -v "$ARTIFACT_DIR/runtime-results:/results" "$CANDIDATE_IMAGE" python /verify_runtime.py record /results/baseline.json --require-video --http-base http://release-web:8080 > "$ARTIFACT_DIR/runtime-before.log" 2>&1
