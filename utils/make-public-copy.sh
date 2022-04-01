set -ex

back_to_master() {
    git checkout master
    git submodule update --init --recursive
}
trap back_to_master EXIT

git branch -D public || true
git checkout -b public

# Remove unneeded files
for f in exp utils .github .gitmodules ; do
    rm -rf $f
    git add $f
done

# Converting submodules into plain files as Anonymous GitHub does not support them yet
# https://github.com/tdurieux/anonymous_github/issues/34
for f in $(find -type f -name .git); do
    folder=${f/.git}
    git rm --cached $folder
    rm -f $f
    if [[ $folder != *TI-DSPLib* && $folder != *ARM-CMSIS* ]]; then
        git add $folder
    else
        rm -rf $folder
    fi
done

git commit -m "Public copy"
