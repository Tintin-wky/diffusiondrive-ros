docker start diffusiondrive 2>/dev/null
docker exec -it diffusiondrive /bin/zsh -c "cd ~/diffusiondrive && source ../catkin_ws/devel/setup.zsh && python3 diffusiondrive_node.py"