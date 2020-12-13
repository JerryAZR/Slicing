echo "docker cp $(docker container ls -lq):$1 $2"
docker cp $(docker container ls -lq):$1 $2
