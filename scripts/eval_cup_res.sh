#!/bin/bash

CURRENT_DIR=`pwd`


usage() {
  echo "Usage: ${0} [-f|--filepath]" 1>&2
  exit 1 
}

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    -f|--filepath)
      FILEPATH=${2}
      shift 2
      ;;
    *)
      usage
      shift
      ;;
  esac
done

function eval() {
    SCRIPT_PATH="src/evaluator/eval_cup.py"
    python $SCRIPT_PATH \
        --filepath $FILEPATH
}

eval;