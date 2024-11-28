 #!/bin/bash
 
get_location() {
  local method=$1
  local dir=$2
  read -p "Do you want to include $method? (1 for \"Yes\"): " include
  if [[ "$include" == "1" ]]; then
    read -p "Enter the name of the experiment: " log_dir
    echo "$dir/$log_dir"
  else
    echo ""
  fi
}

read -p "Enter the name of the plot (output PDF file): " plot_name
read -p "Enter the website name: " site_name

locations=()

sb_location=$(get_location "SB crawler" "crawlers/auer_crawler/output")
focused_location=$(get_location "focused crawler" "crawlers/focused_online_crawler/output")
offline_dom_location=$(get_location "offline-DOM paths crawler" "crawler/dom_off_online_crawler/output")
bfs_location=$(get_location "BFS crawler" "crawlers/bfs_online_crawler/output")
dfs_location=$(get_location "DFS crawler" "crawlers/dfs_online_crawler/output")
random_location=$(get_location "random crawler" "crawlers/random_online_crawler/output")

[[ -n "$sb_location" ]] && locations+=("$sb_location")
[[ -n "$focused_location" ]] && locations+=("$focused_location")
[[ -n "$offline_dom_location" ]] && locations+=("$offline_dom_location")
[[ -n "$bfs_location" ]] && locations+=("$bfs_location")
[[ -n "$dfs_location" ]] && locations+=("$dfs_location")
[[ -n "$random_location" ]] && locations+=("$random_location")

locations_list="$(IFS=','; echo ${locations[*]})"

python3 graphical_results/generate_plots.py "$plot_name" "$site_name" "$locations_list"
