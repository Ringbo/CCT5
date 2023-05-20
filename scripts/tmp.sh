#!/bin/bash
 # set default value for eval variable
eval="default"
 # check if -e flag is provided
while getopts ":e:" opt; do
  case $opt in
    e)
      # if -e flag is provided with a parameter, set eval variable to parameter
      eval="$OPTARG"
      ;;
    \?)
      # if invalid option is provided, print error message and exit
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      # if -e flag is provided without a parameter, set eval variable to true
      eval="true"
      ;;
  esac
done
 # output value of eval variable
echo "Eval parameter: $eval"