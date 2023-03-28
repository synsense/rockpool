#
# Build a testing environment for Rockpool, as well as a base installation container 
#

# - Build and push the test container for spinystellate
#docker login spinystellate.office.synsense.ai:5051
#docker build -t spinystellate.office.synsense.ai:5051/research/rockpool/test:latest test
#docker push spinystellate.office.synsense.ai:5051/research/rockpool/test

# - Build and push the test container to gitlab.com
docker login registry.gitlab.com
docker build -t registry.gitlab.com/synsense/rockpool/test:newest test
docker push registry.gitlab.com/synsense/rockpool/test

# - Build and push the deploy container
# docker build -t spinystellate.office.synsense.ai:5051/research/rockpool/deploy:latest deploy
# docker push spinystellate.office.synsense.ai:5051/research/rockpool/deploy
