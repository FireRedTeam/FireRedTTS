#! /usr/bin/env python3

import argparse
import contextlib
from distutils.log import debug
import os
from pickle import BUILD
import re
import shlex
import shutil
import subprocess
import sys
import multiprocessing
import json
import datetime


now = datetime.datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

def get_shell_result(cmd):
    with os.popen(cmd) as f:
        result = f.readlines()[0].strip()
        return result


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--push", action='store_true',
                    help="push docker hub")
parser.add_argument("-c", "--use_cache",
                    action='store_true', help="use docker cache")
parser.add_argument("--notar",
                    action='store_true', help="no tar photo.tar.gz")
args = parser.parse_args()


use_docker_cache = True if args.use_cache else False
push_docker_hub = True if args.push else False

repo_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tar_src_cmd = "cd .."
tar_src_cmd += f" && tar zcf {repo_name}.tar.gz assets docker configs examples fireredtts pretrained_models tools *.py"
tar_src_cmd += f" && mv {repo_name}.tar.gz docker/ && cd docker/"

if not args.notar:
    os.system(tar_src_cmd)

DK_HUB = "docker-reg.devops.xiaohongshu.com/media"

git_branch_ref = get_shell_result("git rev-parse --abbrev-ref HEAD")
git_branch = os.environ.get("CI_COMMIT_REF_SLUG", default=git_branch_ref)
git_commit = get_shell_result("git rev-parse --short HEAD")

PROD_VERSION = git_branch+"_" + git_commit + "_" + dt_string

print("git branch: {}, git commit: {}".format(git_branch, git_commit))

build_cmd = "docker build"
if not use_docker_cache:
    build_cmd += " --no-cache "

image_name = f"{repo_name}:" + PROD_VERSION
build_cmd += " -t " + image_name
build_cmd += f" --build-arg repo_name={repo_name} --force-rm -f Dockerfile ."

print("build cmd: {}".format(build_cmd))
ret = os.system(build_cmd)
if ret != 0:
    print("build image failed!")
    os.abort()

hub_image_name = DK_HUB + "/" + image_name
tag_cmd = "docker tag " + image_name + " " + hub_image_name
push_cmd = "docker push " + hub_image_name

login_cmd = "docker login -u zijing -p 12345xhS  docker-reg.devops.xiaohongshu.com/media"

if push_docker_hub:
    os.system(tag_cmd)
    loginret = os.system(login_cmd)
    ret = os.system(push_cmd)
    print("docker push: {0}, result: {1}".format(
        push_cmd, "success" if ret == 0 else "failed"))
    print("Build description:{0}".format(hub_image_name))
