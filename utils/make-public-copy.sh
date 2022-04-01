#!/bin/bash
set -ex

back_to_master() {
    git checkout master
    git reset .
    git checkout .
    git submodule update --init --recursive
}
trap back_to_master EXIT

git branch -D public || true
git checkout -b public

# Converting submodules into plain files as Anonymous GitHub does not support them yet
# https://github.com/tdurieux/anonymous_github/issues/34
gitmodules=()
for f in $(find -type f -name .git) ; do
    folder=${f/.git}
    # Ignore possible errors due to nested submodules
    git rm --cached $folder || true
    rm -f $f
    gitmodules+=($folder)
done

declare -p gitmodules

for folder in "${gitmodules[@]}" ; do
    if [[ $folder != *TI-DSPLib* && $folder != *ARM-CMSIS* ]]; then
        git add $folder
    else
        rm -rf $folder
    fi
done

# Remove unneeded files
for f in dnn-models/parse_model dnn-models/*/README.md exp tools/{*.py,.gitmodules,README.md} tools/ext_fram/{README.md,qspiFRAM.*} utils .github .gitmodules ; do
    rm -rf $f
    git add $f
done

git commit -m "Public copy"
