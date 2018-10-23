function results=run_PD(seq, res_path, bSaveImage)
%PeakDifference

close all;

results = vivid_trackers(seq, res_path, bSaveImage, 4);
