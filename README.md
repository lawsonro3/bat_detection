## Bat Detection

# Repository structure

bat_vids: A few videos of bats, for testing

frames:  
    - frames from some of the bat_vids videos, used for intermediate analysis  
    - fft_test_final: frames used for final testing from fft analysis

output:  
    - fft_test_final: data output from final fft analysis, as well as ROI location info  
    - ssim_nccorr_compare: data output from determination of ssim vs. nccorr  

python_scripts:  
    - fft: basic fft scripts  
    - fft_test_final: scripts for final fft comparison procedure (save ROI info first, then conduct comparisons)  
    - practice: initial/intermediate comparison scripts, practice scripts, as well as input file from intermediate analysis  
    - video_reading: basic scripts to open, read, convert videos to frames
