#!/bin/bash

function setup_repo() {
    cd myParser;
    bash myParser/build.sh;
    cd ../
}

function create_and_activate() {
    conda create --name cct5 python=3.8;
    conda activate cct5;
}

function install_deps() {
    conda install torch=2.0.0+cu117 torchvision=0.15.1 torchaudio cudatoolkit=11.7 -c pytorch -c conda-forge
    conda install datasets==1.16.1 -c conda-forge
    conda install transformers==4.21.1 -c conda-forge
    conda install tensorboard==2.12.2 -c conda-forge
    pip install tree-sitter==0.19.1;
    pip install nltk=3.8.1;
    pip install scipy=1.10.1;
}

create_and_activate;
install_deps;
setup_repo;
