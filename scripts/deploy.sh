SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
VERSION_FILE=$SCRIPTPATH/../simplenn/__init__.py

# Run Quality Control
cd $SCRIPTPATH/
if ! bash run_qa.sh;
then
    echo "<DEPLOY> Early exit | QA failed"
    exit 1
fi

# Extract Release Version
RELEASE_VERSION=$(cat $VERSION_FILE | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+')

# Commit to local repo
cd $SCRIPTPATH/..
git add . && git commit -m $RELEASE_VERSION

# Push to remote repo
git push origin master

# publish to PyPi
cd $SCRIPTPATH/
bash pypi.sh