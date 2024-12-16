for dir in lab[1-9]; do
    if [ -d "$dir" ]; then
        new_dir=$(printf "lab%02d" "${dir#lab}")
        echo $new_dir
        # Rename the directory
        mv "$dir" "$new_dir"
    fi
done