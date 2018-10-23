function results=run_TM(seq, res_path, bSaveImage)
%TemplateMatch

close all;

results = vivid_trackers(seq, res_path, bSaveImage, 1);
