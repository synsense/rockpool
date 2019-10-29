#!/bin/bash

array=($(find ./dist/ -name "rockpool*" ))

rezult_pip=($(/usr/bin/pip hash "${array[@]}"))

file_name=${array[@]}
file_name=${file_name#*dist/}

str=${rezult_pip[@]}
sha_with_tag=${str#*=}
sha=${sha_with_tag#*:}

echo ${file_name}
echo ${sha}
