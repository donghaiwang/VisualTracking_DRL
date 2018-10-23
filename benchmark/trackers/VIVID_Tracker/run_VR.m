function results=run_VR(seq, res_path, bSaveImage)
%VarianceRatio

close all;

results = vivid_trackers(seq, res_path, bSaveImage, 3);
