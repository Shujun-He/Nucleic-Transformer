for i in {1..3};do
    mkdir test$i
    cp template/* test$i
    cd test$i
    ./run.sh
    cd ..
done    