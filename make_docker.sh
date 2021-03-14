cp requirements.yml docker
docker login spinystellate.office.synsense.ai:5051
docker build -t spinystellate.office.synsense.ai:5051/research/rockpool .docker
docker push spinystellate.office.synsense.ai:5051/research/rockpool
