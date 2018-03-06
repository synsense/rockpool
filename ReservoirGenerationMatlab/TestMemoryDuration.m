% TestMemoryDuration
% Usage: TestMemoryDuration(mfW, vfBias, fhInput, tLimit, vtPerturb <, vfTau, fZeta, tNoiseFrameDur>)

function TestMemoryDuration(mfW, vfBias, fhInput, tLimit, vtPerturb, varargin)

% - Check arguments
if ~exist('fhInput', 'var') || isempty(fhInput)
   fhInput = @(t)0;
end

% - Run non-perturbed stimulation
nNetSize = size(mfW, 1);
rng(10, 'twister');
[vt, mfX] = SimulateLT(mfW, vfBias, fhInput, tLimit(1), varargin{:});

% - Set up perturbation
fhPert = @(t)[(t >= vtPerturb(1)) .* (t <= vtPerturb(2)); zeros(nNetSize-1, 1)];

% - Run perturbed simulation
rng(10, 'twister');
[vtP, mfXP, vfEigs] = SimulateLT(mfW, vfBias, @(t)fhInput(t) + fhPert(t), tLimit(1), varargin{:});

if (numel(tLimit) > 1)
   vbInclude = vt > tLimit(2);
   vbIncludeP = vtP > tLimit(2);
else
   vbInclude = true(size(vt));
   vbIncludeP = true(size(vtP));
end

vbPerturb = (vtP >= vtPerturb(1)) & (vtP <= vtPerturb(2));

% - Visualise result of perturbation
figure;
subplot(2, 1, 1);
plot(vt(vbInclude), mfX(vbInclude, 1:10:end));
ylabel('x');
title('Reservoir activity');

subplot(2, 1, 2);
plot(vt(vbInclude), mfX(vbInclude, 2:10:end) - interp1(vtP, mfXP(:, 2:10:end), vt(vbInclude)));
vfYLim = ylim;
hold all;
plot(vtPerturb, vfYLim(1) * [1 1], 'k-', 'LineWidth', 4);

ylabel('x-x^''');
title('Reservoir activity');
xlabel('Time');

figure;
plot3(mfX(vbInclude, 2), mfX(vbInclude, 3), vt(vbInclude));
hold all;
plot3(mfXP(vbIncludeP, 2), mfXP(vbIncludeP, 3), vtP(vbIncludeP), '--');
plot3(mfXP(vbPerturb, 2), mfXP(vbPerturb, 3), vtP(vbPerturb), 'k-', 'LineWidth', 4);
axis vis3d;
camproj perspective;
zlabel('Time');
xlabel('x_2');
ylabel('x_3');
title('Reservoir state (2D x t)');

figure;
plot3(mfX(vbInclude, 2), mfX(vbInclude, 3), mfX(vbInclude, 4));
hold all;
plot3(mfXP(vbIncludeP, 2), mfXP(vbIncludeP, 3), mfXP(vbIncludeP, 4), '--');
plot3(mfXP(vbPerturb, 2), mfXP(vbPerturb, 3), mfXP(vbPerturb, 4), 'k-', 'LineWidth', 4);
axis vis3d;
camproj perspective;
xlabel('x_2');
ylabel('x_3');
zlabel('x_4');
title('Reservoir state (3D)');

figure;
plot(vt(vbInclude), mfX(vbInclude, 2));
hold all;
plot(vtP(vbIncludeP), mfXP(vbIncludeP, 2), '--');
plot(vtP(vbPerturb), mfXP(vbPerturb, 2), 'k-', 'LineWidth', 4);
xlabel('Time');
ylabel('x_2');
title('Reservoir activity (x_2)');

% - Generate state vectors (3D)
mfState = [mfX(:, 2:4) [zeros(1, 3); diff(mfX(:, 2:4), 1)]];
mfStateP = [mfXP(:, 2:4) [zeros(1, 3); diff(mfXP(:, 2:4), 1)]];

mfState = mfState(vbInclude, :);
mfStateP = mfStateP(vbIncludeP, :);

% - Compute minimum state distance over time
vfStateDists = pdist2(mfState, mfStateP, 'cityblock', 'smallest', 1);

figure
plot(vtP(vbIncludeP), vfStateDists);
hold all;
plot(vtPerturb, [0 0], 'k-', 'LineWidth', 4);
xlabel('Time');
ylabel('s-s^''');
title('Reservoir state distance');

figure;
PlotEigenspectrum(vfEigs);
title('Eigenspectrum');

