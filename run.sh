echo "Starting batch experiments..."dsds
for gt in ./ground_truth_images/*/flow*.flo; do
    # Extract the folder name (e.g., Venus, Dimetrodon, etc.)
    folder=$(echo "$gt" | cut -d'/' -f3)
    
    # Extract the frame number from the filename "flow10.flo"
    base=$(basename "$gt")
    frame_num=${base#flow}       # remove prefix 'flow'
    frame_num=${frame_num%.flo}   # remove suffix '.flo'
    
    # Construct image paths.
    image1=./images/"$folder"/frame"$frame_num".png
    # For image2, assume the next frame (frame number + 1)
    next_frame=$((10#$frame_num + 1))
    image2=./images/"$folder"/frame"$next_frame".png

    # Check if both image files exist.
    if [[ -f "$image1" && -f "$image2" ]]; then
        echo "Running experiment for folder '$folder':"
        echo "  image1: $image1"
        echo "  image2: $image2"
        echo "  gt file: $gt"
        
        # Run the driver with the desired flags.
        python driver.py "$image1" "$image2" "$gt" --use_affine True --log_results True
        
        # Optional: Pause briefly between runs.
        # sleep 2
    else
        echo "Missing image files for folder '$folder' with gt file $gt. Skipping run."
    fi
done