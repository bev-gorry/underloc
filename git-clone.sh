#!/bin/bash

get_git_url() {
  local git_profile=$1
  local git_repo=$2
  local ssh=$3

  if [ "$ssh" = true ]; then
    echo "git@github.com:$git_profile/$git_repo.git"
  else
    echo "https://github.com/$git_profile/$git_repo.git"
  fi
}

git_profile=$1
git_repo=$2
baselines_folder="$3/$git_repo"
ssh_flag=$4

if [ -d "$baselines_folder" ]; then
  exit 0
fi

if [ "$ssh_flag" = "--ssh" ]; then
  ssh=true
else
  ssh=false
fi

git_url=$(get_git_url ${git_profile} ${git_repo} ${ssh})
git clone --recurse-submodules ${git_url} ${baselines_folder}
