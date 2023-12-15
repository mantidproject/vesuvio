#! /bin/bash
set -ex
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
./config.sh --url https://github.com/mantidproject/vesuvio --token AWZQ65UVPZUA3LFSZDMHE4DFPSK5O
./run.sh