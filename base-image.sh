GPU_REGISTRY="192.168.0.50:5100"
TAG="${GPU_REGISTRY}/penguins-base:0.1"

echo "Build Dockerfile"
docker build -t ${TAG} -f base-image/Dockerfile base-image/.

echo "Push to GPU Registry"
docker push ${TAG}
