#!/bin/bash

array=($(find ./dist/ -name "rockpool*tar.gz" ))

result_pip=($(pip hash "${array[@]}"))

file_name=${array[@]}
file_name=${file_name#*dist//}

str=${result_pip[@]}
sha_with_tag=${str#*=}
sha=${sha_with_tag#*:}

echo ${file_name}
echo ${sha}

version=$(python -c "exec(open('rockpool/version.py').read()); print(__version__)")

echo ${version}

git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/ai-cortex/staged-recipes.git staged-recipes

cd ./staged-recipes

git checkout master

cd ./recipes/rockpool

cp ../../LICENSE .

git config user.email "info@synsense.ai"
git config user.name ${GITHUB_USER}

echo "{% set name = \"rockpool\" %}" > meta.yaml
echo "{% set version = \"${version}\" %}" >> meta.yaml
echo "" >> meta.yaml
echo "package:" >> meta.yaml
echo "  name: {{ name|lower }}" >> meta.yaml
echo "  version: {{ version }}" >> meta.yaml
echo "" >> meta.yaml
echo "source:" >> meta.yaml
echo "  url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz" >> meta.yaml
echo "  sha256: ${sha}" >> meta.yaml
echo "" >> meta.yaml
echo "build:" >> meta.yaml
echo "  noarch: python" >> meta.yaml
echo "  number: 0" >> meta.yaml
echo "  script: \"{{ PYTHON }} -m pip install . -vv\"" >> meta.yaml
echo "" >> meta.yaml
echo "requirements:" >> meta.yaml
echo "  host:" >> meta.yaml
echo "    - python >=3.6" >> meta.yaml
echo "    - pip" >> meta.yaml
echo "    - setuptools" >> meta.yaml
echo "  run:" >> meta.yaml
echo "    - numpy" >> meta.yaml
echo "    - scipy" >> meta.yaml
echo "    - numba" >> meta.yaml
echo "    - python >=3.6" >> meta.yaml
echo "" >> meta.yaml
echo "test:" >> meta.yaml
echo "  imports:" >> meta.yaml
echo "    - rockpool" >> meta.yaml
echo "" >> meta.yaml
echo "about:" >> meta.yaml
echo "  home: https://rockpool.ai" >> meta.yaml
echo "  license: AGPL-3.0-only" >> meta.yaml
echo "  license_family: AGPL" >> meta.yaml
echo "  license_file: LICENSE" >> meta.yaml
echo "  summary: 'Python package for developing, simulating and training spiking neural networks, and deploying on Neuromorphic hardware'" >> meta.yaml
echo "" >> meta.yaml
echo "  description: |" >> meta.yaml
echo "    Rockpool is a Python package for working with dynamical neural network architectures, particularly for designing event-driven networks for Neuromorphic computing hardware. Rockpool provides a convenient interface for designing, training and evaluating recurrent networks, which can operate both with continuous-time dynamics and event-driven dynamics." >> meta.yaml
echo "    Rockpool is an open-source project managed by SynSense" >> meta.yaml
echo "  doc_url: https://rockpool.ai" >> meta.yaml
echo "  dev_url: https://gitlab.com/SynSense/rockpool" >> meta.yaml
echo "" >> meta.yaml
echo "extra:" >> meta.yaml
echo "  recipe-maintainers:" >> meta.yaml
echo "    - DylanMuir" >> meta.yaml

cd ..

git add -A
git commit -m "Staging conda-forge for ${file_name}"
git push origin
git request-pull master https://github.com/conda-forge/rockpool-feedstock

cd ..

rm -rf ./staged-recipes
