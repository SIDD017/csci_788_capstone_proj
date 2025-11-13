export PYTHONIOENCODING=UTF-8
echo "Starting batch experiments..."
for gt in ./ground_truth_images/*/flow*.flo; do
    # Extract the folder name (e.g., Venus, Dimetrodon, etc.)
    folder=$(echo "$gt" | cut -d'/' -f3)
    
    # Extract the frame number from the filename "flow10.flo"
    base=$(basename "$gt")
    frame_num=${base#flow}
    frame_num=${frame_num%.flo}
    
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
        
        python main.py --image1_path "$image1" --image2_path "$image2" --gtimage_path "$gt" --experiment_name "refine_experiment" --run_name "run_$folder"
    fi
done