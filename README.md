## Bat Detection

Repository structure

bat_vids: Test videos of bats

frames:
    - frames from some of the bat_vids videos, used for intermediate analysis
    - clear_background: frames from different categories of videos, used for intermediate analysis (not used for final analyses)
    - final: frames actually used for final comparisons
    - test: test frames

output:
    - figs: output from intermediate analysis
    - final: data output from final analysis, as well as ROI location info
    - output.csv: data output from intermediate analysis

python_scripts:
    - fft: fft practice scripts
    - practice: initial/intermediate comparison scripts, practice scripts, as well as input file from intermediate analysis
    - video_reading: scripts to open, read, convert videos to frames
    - compare_diff_categories.py, compare_same_categories.py, save_roi.py: scripts for final comparison procedure (save ROI info first, then conduct comparisons)
