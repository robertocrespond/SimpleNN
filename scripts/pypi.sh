SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/..
rm -rf dist
python3 setup.py sdist bdist_wheel
# python3 -m twine upload --repository testpypi dist/*
python3 -m twine upload dist/*