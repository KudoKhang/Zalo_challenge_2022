# Prepare data
```bash
# Generate depth map by TDDFA
python3 TDDFA/process_depth_v3.py

# Visualize and check balance depth and live
python3 TDDFA/check_liveand_depth.py

# Generate train_list.txt & test_list.txt (need 1 live and 1 spoof)
python3 dataset/gen_train_test.txt.py

# Train. Replace train_list & test_list.txt
python3 CDCN/train_private
```
