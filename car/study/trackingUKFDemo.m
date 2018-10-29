%% Constant-Velocity Unscented Kalman Filter
filter = trackingUKF(@constvel, @cvmeas, [0;0;0;0], 'Alpha', 1e-2);
means = [1;1;0];
[xpred, Ppred] = predict(filter);
[xcorr, Pcorr] = correct(filter, means);
[xpred, Ppred] = predict(filter);
[xpred, Ppred] = predict(filter)