#!/bin/bash

set -eux

cd $(dirname $0)/../

colcon build

set +eux
source install/setup.bash
set -eux

export PYTHONPATH=${PYTHONPATH}:$(pwd)/../

ros2 run orienter_net_pkg orienter_net_node \
    --ros-args --param dummy:=value
