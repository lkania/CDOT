#!/bin/bash

# load servers
. servers.sh

command="free -m | awk 'NR==2{printf \"Memory Usage\t%s/%sMB\t\t%.2f%\n\", \$3, \$2, \$3/\$2*100}';"
command+="top -n 1 | grep load | awk '{printf \"CPU Load\t%.2f\t\t\t%.2f\n\", \$(NF-2), \$(NF-1)}';"
command+="exit"

for server in "${servers[@]}"; do
  printf '=%.0s' {1..50};
  printf '\n'
  echo $server
  ssh $server -q -t "$command"
done