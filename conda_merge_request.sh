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

version=$(echo "${file_name}" | sed -n "s/^.*-\s*\(\S*\).*.tar.gz$/\1/p")

git clone https://ai-cortex:aicortexAtgithub2019@github.com/ai-cortex/staged-recipes.git staged-recipes

cd ./staged-recipes

git checkout rockpool

cd ./recipes/rockpool

echo "{% set name = \"rockpool\" %}" > meta.yaml
echo "{% set version = \"${version}\" %}" >> meta.yaml
echo "" >> meta.yaml
echo "package:" >> meta.yaml
echo "  name: {{ name|lower }}" >> meta.yaml
echo "  version: {{ version }}" >> meta.yaml
echo "" >> meta.yaml
echo "source:" >> meta.yaml
echo "  url: https://pypi.org/project/rockpool/packages/${version}/${file_name}" >> meta.yaml
echo "  sha256: ${sha}" >> meta.yaml
echo "" >> meta.yaml
echo "build:" >> meta.yaml
echo "  noarch: python" >> meta.yaml
echo "  number: 0" >> meta.yaml
echo "  skip: True  # [py<36]" >> meta.yaml
echo "  script: \"{{ PYTHON }} -m pip install . -vv\"" >> meta.yaml
echo "" >> meta.yaml
echo "requirements:" >> meta.yaml
echo "  build:" >> meta.yaml
echo "  host:" >> meta.yaml
echo "    - python" >> meta.yaml
echo "    - pip" >> meta.yaml
echo "  run:" >> meta.yaml
echo "    - python" >> meta.yaml
echo "" >> meta.yaml
echo "test:" >> meta.yaml
echo "  imports:" >> meta.yaml
echo "    - rockpool" >> meta.yaml
echo "    - rockpool.tests" >> meta.yaml
echo "" >> meta.yaml
echo "about:" >> meta.yaml
echo "  home: https://gitlab.com/ai-ctx/rockpool" >> meta.yaml
echo "  license: AGPL-3.0" >> meta.yaml
echo "  license_family:" >> meta.yaml
echo "  license_file: LICENSE" >> meta.yaml
echo "  summary: 'Python package for developing, simulating and training spiking neural networks, and deploying on Neuromorphic hardware'" >> meta.yaml
echo "" >> meta.yaml
echo "  description: |" >> meta.yaml
echo "    Rockpool is a Python package for working with dynamical neural network architectures, particularly for designing event-driven networks for Neuromorphic computing hardware. Rockpool provides a convenient interface for designing, training and evaluating recurrent networks, which can operate both with continuous-time dynamics and event-driven dynamics." >> meta.yaml
echo "    Rockpool is an open-source project managed by aiCTX AG" >> meta.yaml
echo "  doc_url: https://rockpool.readthedocs.io/" >> meta.yaml
echo "  dev_url: https://github.com/rockpool/rockpool" >> meta.yaml
echo "" >> meta.yaml
echo "extra:" >> meta.yaml
echo "  recipe-maintainers:" >> meta.yaml
echo "    - Dylan Richard Muir" >> meta.yaml
echo "    - Felix Bauer" >> meta.yaml
echo "    - Marco Reato" >> meta.yaml
echo "    - Philipp Weidel" >> meta.yaml
echo "    - Dacian Herbei" >> meta.yaml
echo "" >> meta.yaml

cd ../..

git add -A
git commit -m "version ${file_name}"
git push origin

cd ..

rm -rf ./staged-recipes