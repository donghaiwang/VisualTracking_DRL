function results=run_MS(seq, res_path, bSaveImage)
%MeanShift

close all;

results = vivid_trackers(seq, res_path, bSaveImage, 2);
