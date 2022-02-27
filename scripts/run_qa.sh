#!/bin/bash
##################################################################################################
# Custom QA Pipeline
# 
# Run every *module* on tests and on package .py files.
#   Sucessful run return status code: 0
#   Failed run return status code: 1
# 
# Modules
# - reorder-python-imports
# - black
# - flake8
# - mypy
# - pytest & coverage
# 
##################################################################################################
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/

# Configuration
MIN_TEST_COVERAGE=0 # percentage of test coverage needed
PACKAGE_PATH=$SCRIPTPATH/../simplenn
TESTS_PATH=$SCRIPTPATH/../tests

##################################################################################################
# [reorder-python-imports] 
##################################################################################################
# Package modules
echo "[reorder-python-imports] $PACKAGE_PATH/"
IFS=$'\n'; set -f
for f in $(find $PACKAGE_PATH -name '*.py'); 
    do reorder-python-imports "$f"; 
done
unset IFS; set +f

# Testing modules
echo "[reorder-python-imports] $TESTS_PATH/"
IFS=$'\n'; set -f
for f in $(find $TESTS_PATH -name '*.py'); 
    do reorder-python-imports "$f"; 
done
unset IFS; set +f

##################################################################################################
# [black] 
##################################################################################################
# Package modules
echo "[black] $PACKAGE_PATH/"
black $PACKAGE_PATH

# Testing modules
echo "[black] $TESTS_PATH/"
black $TESTS_PATH

##################################################################################################
# [flake8] 
##################################################################################################
# Package modules
echo "[flake8] $PACKAGE_PATH/"
if ! flake8 $PACKAGE_PATH;
then
    echo "Early exit: flake8 exited with code != 0"
    exit 1
fi

# Testing modules
echo "[flake8] $TESTS_PATH/"
if ! flake8 $TESTS_PATH;
then
    echo "Early exit: flake8 exited with code != 0"
    exit 1
fi

##################################################################################################
# [mypy] 
##################################################################################################
# Package modules
echo "[mypy] $PACKAGE_PATH/"
if ! mypy $PACKAGE_PATH;
then
    echo "Early exit: mypy exited with code != 0"
    exit 1
fi

# Testing modules
echo "[mypy] $TESTS_PATH/"
if ! mypy $TESTS_PATH;
then
    echo "Early exit: mypy exited with code != 0"
    exit 1
fi

##################################################################################################
# [pytest / coverage] 
##################################################################################################
echo "[pytest / coverage] $PACKAGE_PATH/"
if ! pytest --cov=$PACKAGE_PATH --cov-report term-missing $TESTS_PATH/ --disable-pytest-warnings -v --cov-fail-under=$MIN_TEST_COVERAGE;
then
    echo "Early exit: pytest exited with code != 0"
    exit 1
fi