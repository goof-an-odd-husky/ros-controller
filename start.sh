#!/bin/bash

sim_path="${1:-../husky-sim}"

docker compose -f ${sim_path}/docker-compose.yaml run -v $(pwd):/workspace/goof-an-odd-husky -d --name husky_dev husky_dev
