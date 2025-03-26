docker run --rm \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v /inputPaths.txt:/app/custom_inputPaths.txt \
 -v /TestOutput:/app/TestOutput \
 --gpus all \
 runmodel -i /app/custom_inputPaths.txt
