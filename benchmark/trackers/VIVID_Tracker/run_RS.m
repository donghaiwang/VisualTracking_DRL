function results=run_RS(seq, res_path, bSaveImage)
%RatioShift

close all;

results = vivid_trackers(seq, res_path, bSaveImage, 5);
