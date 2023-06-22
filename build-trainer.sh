GPU_REGISTRY="192.168.0.50:5100"
TAG="${GPU_REGISTRY}/penguins-trainer:0.1"

echo "Build Dockerfile"
docker build -t ${TAG} -f pipeline/trainer-docker/Dockerfile pipeline/trainer-docker/.

echo "Push to GPU Registry"
docker push ${TAG}