function TestMemoryDuration(mfW, vfBias, fhInput, tLimit, vtPerturb, varargin)

% TestMemoryDuration
% Usage: TestMemoryDuration(mfW, vfBias, fhInput, tLimit, vtPerturb <, vfTau, fZeta, tNoiseFrameDur>)

% - Run non-perturbed stimulation
nNetSize = size(mfW, 1);
rng(10, 'twister');
[vt, mfX] = SimulateLT(mfW, vfBias, fhInput, tLimit(1), varargin{:});

% - Set up perturbation
fhPert = @(t)[(t >= vtPerturb(1)) .* (t <= vtPerturb(2)); zeros(nNetSize-1, 1)];

% - Run perturbed simulation
rng(10, 'twister');
[vtP, mfXP] = SimulateLT(mfW, vfBias, @(t)fhInput(t) + fhPert(t), tLimit(1), varargin{:});

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
plot(vt(vbInclude), mfX(vbInclude, :));
subplot(2, 1, 2);
plot(vt(vbInclude), mfX(vbInclude, 2:end) - interp1(vtP, mfXP(:, 2:end), vt(vbInclude)));

figure;
plot3(mfX(vbInclude, 2), mfX(vbInclude, 3), mfX(vbInclude, 4));
hold all;
plot3(mfXP(vbIncludeP, 2), mfXP(vbIncludeP, 3), mfXP(vbIncludeP, 4));
plot3(mfXP(vbPerturb, 2), mfXP(vbPerturb, 3), mfXP(vbPerturb, 4), 'k-', 'LineWidth', 4);

% - Generate state vectors (3D)
mfState = [mfX(:, 2:4) [zeros(1, 3); diff(mfX(:, 2:4), 1)]];
mfStateP = [mfXP(:, 2:4) [zeros(1, 3); diff(mfXP(:, 2:4), 1)]];

mfState = mfState(vbInclude, :);
mfStateP = mfStateP(vbIncludeP, :);

% - Compute minimum state distance over time
vfStateDists = pdist2(mfState, mfStateP, 'cityblock', 'smallest', 1);

figure
plot(vtP(vbIncludeP), vfStateDists);
