#
# Build a testing environment for Rockpool, as well as a base installation container 
#

# - Login to container repository
# docker login spinystellate.office.synsense.ai:5051
docker login registry.gitlab.com

# - Get the base containiner
docker pull continuumio/miniconda3

# - Build and push the test container
# docker build -t spinystellate.office.synsense.ai:5051/research/rockpool/test:latest test
# docker push spinystellate.office.synsense.ai:5051/research/rockpool/test

docker build -t registry.gitlab.com/synsense/rockpool/test:latest test
docker push registry.gitlab.com/synsense/rockpool/test

# - Build and push the deploy container
# docker build -t spinystellate.office.synsense.ai:5051/research/rockpool/deploy:latest deploy
# docker push spinystellate.office.synsense.ai:5051/research/rockpool/deploy
