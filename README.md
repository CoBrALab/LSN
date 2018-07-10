# LSN
Longitudinal Siamese Network for clinical trajectory prediction 
- Code modules
  - Trajectory prediction 
  (can be used to predict other clinical tasks as well) 
    -  LSN/notebooks/LSN_sim_testcode.ipynb: Stand-alone notebook for testing LSN with simulated data
    -  LSN/lib/lsn.py: LSN model class and useful functions for training and testing
    -  LSN/notebooks/run_lsn.ipynb: notebook to train and test LSN model with real data
    
  - Trajectory modeling  
    -  LSN/notebooks/model_trajectories.ipynb: Clustering code for modeling longitudinal clinical trajectories and subsequent assignment to new subjects 
    
- LSN data flow
    -  input (check notebooks for required data shapes) 
        - MR: baseline + follow-up (e.g. 78x2 AAL CT values) 
        - aux: apoe4 status + clinical scores (baseline + follow-up) + demographics (optional)      
    -  output (one-hot) 
        - labels: trajectory / Dx / Px labels (binary and multiclass are supported) 
        
- Legacy dir has older code version along with useful notebook for mapping vertex-wise CIVET data into an atlas based ROIs. 
  
- Prereqs 
  - python3.5+
  - tensorflow-gpu 1.4.1 (conda: conda install -c anaconda tensorflow-gpu)
  - sklearn
  - pandas
  - seaborn 

  
