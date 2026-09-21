#!/usr/bin/env bash
set -euo pipefail
ARTIFACT_DIR=/tmp/sfn-release-artifacts
CANDIDATE_IMAGE=scalarforensic:release-check-20260908
CHECK_SCRIPT=/home/user01/Schreibtisch/gitea/ScalarForensic/scripts/verify_case_state.py
docker stop sfn-release-web-source sfn-release-db-source
docker cp sfn-release-web-source:/app/data "$ARTIFACT_DIR/backup/app-data"
docker cp sfn-release-db-source:/qdrant/storage "$ARTIFACT_DIR/backup/qdrant-storage"
python3 "$CHECK_SCRIPT" capture "$ARTIFACT_DIR/backup/app-data" "$ARTIFACT_DIR/app-state.sha256.json"
python3 "$CHECK_SCRIPT" capture "$ARTIFACT_DIR/backup/qdrant-storage" "$ARTIFACT_DIR/qdrant-state.sha256.json"
cp -a "$ARTIFACT_DIR/backup/app-data/." "$ARTIFACT_DIR/restored-state/"
python3 "$CHECK_SCRIPT" verify "$ARTIFACT_DIR/restored-state" "$ARTIFACT_DIR/app-state.sha256.json"
docker run --rm --network none -v "$ARTIFACT_DIR/restored-state:/state" alpine:latest chown -R 10001:10001 /state
docker network create --internal sfn-release-restore-20260908
docker create --name sfn-release-db-restored --network sfn-release-restore-20260908 --network-alias release-qdrant qdrant/qdrant:v1.19.0
docker cp "$ARTIFACT_DIR/backup/qdrant-storage/." sfn-release-db-restored:/qdrant/storage
docker cp sfn-release-db-restored:/qdrant/storage "$ARTIFACT_DIR/restored-db-before-start"
python3 "$CHECK_SCRIPT" verify "$ARTIFACT_DIR/restored-db-before-start" "$ARTIFACT_DIR/qdrant-state.sha256.json"
docker start sfn-release-db-restored
docker run -d --name sfn-release-web-restored --network sfn-release-restore-20260908 --network-alias release-web --env-file "$ARTIFACT_DIR/runtime.env" -e SFN_DEVICE=cpu -v "$ARTIFACT_DIR/evidence:/images:ro" -v "$ARTIFACT_DIR/restored-state:/app/data" "$CANDIDATE_IMAGE"
for i in $(seq 1 30); do
    if docker exec sfn-release-web-restored python -c 'import urllib.request; assert urllib.request.urlopen("http://localhost:8080/").status == 200' 2>/dev/null; then break; fi
    if [ "$i" = 30 ]; then exit 1; fi
    sleep 2
done
docker run --rm --network sfn-release-restore-20260908 --env-file "$ARTIFACT_DIR/runtime.env" -v "$ARTIFACT_DIR/evidence:/images:ro" -v "$ARTIFACT_DIR/restored-state:/app/data:ro" -v "$ARTIFACT_DIR/verify_runtime.py:/verify_runtime.py:ro" -v "$ARTIFACT_DIR/runtime-results:/results:ro" "$CANDIDATE_IMAGE" python /verify_runtime.py verify /results/baseline.json --require-video --http-base http://release-web:8080
