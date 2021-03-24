#
# Build a testing environment for Rockpool, as well as a base installation container 
#

# - Login to container repository
docker login spinystellate.office.synsense.ai:5051

# - Build and push the test container
docker build -t spinystellate.office.synsense.ai:5051/research/rockpool/test:latest test
docker push spinystellate.office.synsense.ai:5051/research/rockpool/test

# - Build and push the deployed container
# docker build -t spinystellate.office.synsense.ai:5051/research/rockpool/deploy:latest deploy
# docker push spinystellate.office.synsense.ai:5051/research/rockpool/deploy
