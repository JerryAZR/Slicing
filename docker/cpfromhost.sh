echo "docker cp $1 $(docker container ls -lq):$2"
docker cp $1 $(docker container ls -lq):$2
