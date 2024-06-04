 #!/bin/bash
 
get_location() {
  local method=$1
  local dir=$2
  read -p "Do you want to include $method? (1 for \"Yes\"): " include
  if [[ "$include" == "1" ]]; then
    read -p "Enter the log directory for $method: " log_dir
    echo "$dir/$log_dir"
  else
    echo ""
  fi
}

clear
read -p "Enter the name of the plot (output PDF file): " plot_name
read -p "Enter the website name: " site_name

locations=()

sb_location=$(get_location "SB crawler" "auer/logs")
offline_dom_location=$(get_location "offline-DOM paths crawler" "offline_dom/logs")
bfs_location=$(get_location "BFS crawler" "bfs/logs")
dfs_location=$(get_location "DFS crawler" "dfs/logs")
random_location=$(get_location "random crawler" "random/logs")

[[ -n "$sb_location" ]] && locations+=("$sb_location")
[[ -n "$offline_dom_location" ]] && locations+=("$offline_dom_location")
[[ -n "$bfs_location" ]] && locations+=("$bfs_location")
[[ -n "$dfs_location" ]] && locations+=("$dfs_location")
[[ -n "$random_location" ]] && locations+=("$random_location")

locations_list="$(IFS=','; echo ${locations[*]})"

python3 graphical_results/generate_plots.py "$plot_name" "$site_name" "$locations_list"
